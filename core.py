import pandas as pd
import numpy as np


def classify_streams(df):
    df = df.copy()
    df["Tin"] = pd.to_numeric(df["Tin"], errors="coerce")
    df["Tout"] = pd.to_numeric(df["Tout"], errors="coerce")
    df["CP"] = pd.to_numeric(df["CP"], errors="coerce")
    df = df.dropna(subset=["Tin", "Tout", "CP"])
    df["Type"] = df.apply(
        lambda row: "Hot" if row["Tin"] > row["Tout"] else "Cold", axis=1
    )
    return df


def adjust_temperatures(df, delta_tmin):
    delta = delta_tmin / 2
    df = df.copy()
    df["Tin_adj"] = df.apply(
        lambda r: r["Tin"] - delta if r["Type"] == "Hot" else r["Tin"] + delta, axis=1
    )
    df["Tout_adj"] = df.apply(
        lambda r: r["Tout"] - delta if r["Type"] == "Hot" else r["Tout"] + delta, axis=1
    )
    temps = sorted(set(df["Tin_adj"]).union(df["Tout_adj"]), reverse=True)
    intervals = [(temps[i], temps[i + 1]) for i in range(len(temps) - 1)]
    return df, intervals


def calculate_delta_h(df, intervals):
    results = []
    for high, low in intervals:
        hot_cp = 0
        cold_cp = 0
        hot_streams = []
        cold_streams = []
        for _, s in df.iterrows():
            t_high = max(s["Tin_adj"], s["Tout_adj"])
            t_low = min(s["Tin_adj"], s["Tout_adj"])
            if high <= t_high and low >= t_low:
                if s["Type"] == "Hot":
                    hot_cp += s["CP"]
                    hot_streams.append(s["Stream ID"])
                else:
                    cold_cp += s["CP"]
                    cold_streams.append(s["Stream ID"])
        delta_h = (cold_cp - hot_cp) * (high - low)
        results.append({
            "T_high": high,
            "T_low": low,
            "Hot Streams": hot_streams,
            "Cold Streams": cold_streams,
            "ΔH": delta_h,
        })
    return pd.DataFrame(results)


def cascade(delta_h_df):
    rows = []
    cum = 0
    rows.append({"T_high": None, "T_low": None, "Interval": "Start", "Cum ΔH": 0})

    for _, row in delta_h_df.iterrows():
        cum = cum - row["ΔH"]
        rows.append({
            "T_high": row["T_high"],
            "T_low": row["T_low"],
            "Interval": f"{row['T_high']} → {row['T_low']}",
            "Cum ΔH": cum,
        })

    cascade_df = pd.DataFrame(rows)

    min_val = cascade_df["Cum ΔH"].min()
    cascade_df["Adjusted ΔH"] = cascade_df["Cum ΔH"] - min_val

    qh = cascade_df["Adjusted ΔH"].iloc[0]
    qc = cascade_df["Adjusted ΔH"].iloc[-1]

    # Pinch = row where Adjusted ΔH == 0 (excluding start row)
    pinch_candidates = cascade_df.iloc[1:][
        np.isclose(cascade_df.iloc[1:]["Adjusted ΔH"], 0, atol=1e-6)
    ]

    if pinch_candidates.empty:
        # Threshold problem — no true pinch
        pinch_temp = None
        pinch_interval = (None, None)
    else:
        pinch_row = pinch_candidates.iloc[0]
        pinch_temp = pinch_row["T_low"]
        pinch_interval = (pinch_row["T_high"], pinch_row["T_low"])

    return cascade_df, qh, qc, pinch_temp, pinch_interval


def stream_energy_table(df):
    df = df.copy()
    df["ΔT"] = abs(df["Tin"] - df["Tout"])
    df["Heat Duty"] = df["CP"] * df["ΔT"]
    return df[["Stream ID", "Type", "CP", "ΔT", "Heat Duty"]]


def build_stream_interval_tables(df_adj, intervals):
    """
    Returns two DataFrames (hot_table, cold_table).

    Each table has:
      - "Interval"  : "T_high → T_low"
      - one column per stream showing duty Q = CP × ΔT for that interval (0 if stream absent)
      - "Net Duty"  : sum across all streams of that type in the interval
      - "Cumulative Duty" : running total of Net Duty from the top interval downward
    """
    hot_ids  = df_adj[df_adj["Type"] == "Hot"]["Stream ID"].tolist()
    cold_ids = df_adj[df_adj["Type"] == "Cold"]["Stream ID"].tolist()

    hot_rows  = []
    cold_rows = []

    for high, low in reversed(intervals):
        dT = high - low
        interval_label = f"{low} → {high}"

        hot_row  = {"Interval": interval_label}
        cold_row = {"Interval": interval_label}

        for _, s in df_adj.iterrows():
            t_high = max(s["Tin_adj"], s["Tout_adj"])
            t_low  = min(s["Tin_adj"], s["Tout_adj"])
            active = high <= t_high and low >= t_low
            q = round(s["CP"] * dT, 4) if active else 0.0

            if s["Type"] == "Hot":
                hot_row[s["Stream ID"]] = q
            else:
                cold_row[s["Stream ID"]] = q

        # Ensure every stream column exists even if stream not active
        for sid in hot_ids:
            hot_row.setdefault(sid, 0.0)
        for sid in cold_ids:
            cold_row.setdefault(sid, 0.0)

        hot_row["Net Duty"]  = round(sum(hot_row[sid] for sid in hot_ids), 4)
        cold_row["Net Duty"] = round(sum(cold_row[sid] for sid in cold_ids), 4)

        hot_rows.append(hot_row)
        cold_rows.append(cold_row)

    hot_df  = pd.DataFrame(hot_rows,  columns=["Interval"] + hot_ids  + ["Net Duty"])
    cold_df = pd.DataFrame(cold_rows, columns=["Interval"] + cold_ids + ["Net Duty"])

    hot_df["Cumulative Duty"]  = hot_df["Net Duty"].cumsum().round(4)
    cold_df["Cumulative Duty"] = cold_df["Net Duty"].cumsum().round(4)

    return hot_df, cold_df


def build_composite_curves(df_adj, intervals, delta_tmin, qc):
    """
    Returns (hot_points, cold_points).

    Both curves walk low→high temperature (reversed intervals).
    Each active interval contributes ONE point: (cumH_after, t_low_).
    Zero-duty intervals are skipped entirely — no point, no gap, no anchor.
    t_low_:
        hot  = low_adj + delta_tmin/2   (un-shift hot downward correction)
        cold = low_adj - delta_tmin/2   (un-shift cold upward correction)

    Cold H is shifted right by qh (minimum hot utility) so the two curves
    share the same axis and the overlap region is visible.

    Each return value is a list of segments (list of (H, T) tuples).
    A segment break only occurs when a zero-duty interval interrupts an
    otherwise active run — i.e. streams exist above AND below a gap interval.
    """
    delta = delta_tmin/2

    def _build(stream_type, h_offset=0.0):
        segments   = []
        current_seg = None
        cum_h      = 0.0

        # Walk low → high: reversed(intervals) gives (high_adj, low_adj)
        # but we process from lowest T upward, so iterate reversed intervals
        for high_adj, low_adj in reversed(intervals):
            dT     = high_adj - low_adj
            cp_sum = 0.0

            for _, s in df_adj.iterrows():
                s_high = max(s["Tin_adj"], s["Tout_adj"])
                s_low  = min(s["Tin_adj"], s["Tout_adj"])
                if s["Type"] == stream_type and high_adj <= s_high and low_adj >= s_low:
                    cp_sum += s["CP"]

            q = round(cp_sum * dT, 6)

            if q > 1e-9:
                if current_seg is None:
                    # Anchor: point before this interval at current cumH, T = low_adj actual
                    if stream_type == "Hot":
                        t_entry = low_adj + delta
                    else:
                        t_entry = low_adj - delta
                    current_seg = [(h_offset + cum_h, t_entry)]

                cum_h += q

                if stream_type == "Hot":
                    t_exit = high_adj + delta
                else:
                    t_exit = high_adj - delta

                current_seg.append((h_offset + cum_h, t_exit))
            else:
                if current_seg is not None:
                    segments.append(current_seg)
                    current_seg = None
                # H does not advance for zero-duty intervals

        if current_seg is not None:
            segments.append(current_seg)

        return segments

    hot_segments  = _build("Hot",  h_offset=0.0)
    cold_segments = _build("Cold", h_offset=qc)

    return hot_segments, cold_segments


def build_gcc_curve(cascade_df, delta_tmin):
    """
    Grand Composite Curve: plot shifted temperature vs net heat flow.
    Shifted T = actual T + ΔTmin/2 for cold, actual T - ΔTmin/2 for hot.
    Here we use the interval boundary temperatures directly (already shifted).
    """
    rows = cascade_df.dropna(subset=["T_high", "T_low"]).copy()

    # Build (T*, H) pairs: one point per interval boundary
    T_star = []
    H_adj = []

    # Top boundary of first interval
    T_star.append(rows.iloc[0]["T_high"])
    # The adjusted ΔH at the row before the first interval = qh (start row)
    H_adj.append(cascade_df.iloc[0]["Adjusted ΔH"])

    for _, row in rows.iterrows():
        T_star.append(row["T_low"])
        H_adj.append(row["Adjusted ΔH"])

    return H_adj, T_star


def build_hen_grid_data(df, pinch_temp, delta_tmin):
    """
    Returns structured data for the initial HEN grid diagram (no exchangers yet).

    Returns:
        hot_streams: list of dicts with stream info for streams above/at/below pinch
        cold_streams: list of dicts
        pinch: hot-side pinch T (actual)
        pinch_cold: cold-side pinch T (actual)
    """
    df = df.copy()

    pinch = pinch_temp # hot stream pinch boundary (actual)
    pinch_cold = pinch_temp - delta_tmin/2 # cold stream pinch boundary (actual)

    hot_streams = []
    cold_streams = []

    for _, s in df.iterrows():
        t_supply = s["Tin"]
        t_target = s["Tout"]
        stream_type = s["Type"]
        sid = s["Stream ID"]
        cp = s["CP"]
        q = cp * abs(t_supply - t_target)

        entry = {
            "id": sid,
            "CP": cp,
            "Q": q,
            "Tin": t_supply,
            "Tout": t_target,
        }

        if stream_type == "Hot":
            hot_streams.append(entry)
        else:
            cold_streams.append(entry)

    return hot_streams, cold_streams, pinch, pinch_cold

def plot_cascade_diagram(cascade_df, shifted=True):
    """
    Plotly cascade diagram.
 
    Each interval is a box. Inside each box: the cumulative heat flow value.
    Between boxes: a vertical arrow labelled with the interval ΔH.
    QH enters at the top (above the first box), QC exits at the bottom.
    Pinch point is highlighted in red (only in the shifted cascade).
 
    Parameters
    ----------
    cascade_df  : DataFrame from cascade(), contains T_high, T_low,
                  Cum ΔH, Adjusted ΔH columns.
    delta_tmin  : used only to un-shift temperatures for display labels.
    shifted     : if True, use Adjusted ΔH column (all ≥ 0, pinch = 0).
                  if False, use Cum ΔH column (may contain negatives).
    """
    import plotly.graph_objects as go
 
    # ── pull interval rows (exclude the "Start" header row) ──────────────────
    rows = cascade_df.dropna(subset=["T_high", "T_low"]).reset_index(drop=True)
    n = len(rows)
 
    val_col = "Adjusted ΔH" if shifted else "Cum ΔH"
 
    # Values: one per interval boundary (n+1 values for n intervals)
    # Index 0 = top (before first interval), index i+1 = after interval i
    vals = [cascade_df.iloc[0][val_col]] + list(rows[val_col])
 
    # ── layout constants ──────────────────────────────────────────────────────
    box_h    = 0.6    # box height in y-units
    gap      = 0.8    # gap between boxes
    step     = box_h + gap
    box_x0, box_x1 = 0.2, 0.8   # box left/right in [0,1] x space
    box_cx   = (box_x0 + box_x1) / 2
    arrow_x  = box_cx
    label_x  = box_x1 + 0.05
 
    total_h  = n * step + box_h + 1.5   # total figure height in y-units
    y_start  = total_h - 0.5            # y of top of first box
 
    shapes     = []
    annotations = []
 
    # ── QH arrow entering at the top ─────────────────────────────────────────
    qh_val = vals[0]
    qh_y_tip  = y_start + box_h + 0.1
    qh_y_tail = y_start + box_h + 0.7
 
    if qh_val > 1e-6:
        annotations.append(dict(
            x=arrow_x, y=qh_y_tip,
            ax=arrow_x, ay=qh_y_tail,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="red",
            text="", standoff=0,
        ))
        annotations.append(dict(
            x=label_x, y=(qh_y_tip + qh_y_tail) / 2,
            xref="x", yref="y",
            text=f"<b>QH = {qh_val:.2f} kW</b>",
            showarrow=False, font=dict(color="red", size=11),
            xanchor="left",
        ))
 
    # ── draw each box ─────────────────────────────────────────────────────────
    for i, row in rows.iterrows():
        y_top_box = y_start - i * step
        y_bot_box = y_top_box - box_h
        val_inside = vals[i + 1]   # cascade value after this interval
 
        # Is this the pinch interval?
        is_pinch = shifted and abs(val_inside) < 1e-6
 
        fill_color = "rgba(255,200,200,0.6)" if is_pinch else "rgba(173,216,230,0.5)"
        line_color = "red" if is_pinch else "black"
 
        # Box rectangle
        shapes.append(dict(
            type="rect",
            x0=box_x0, x1=box_x1,
            y0=y_bot_box, y1=y_top_box,
            fillcolor=fill_color,
            line=dict(color=line_color, width=1.5),
        ))
 
        # Temperature labels (un-shifted back to actual)
        t_high_ = row["T_high"]   # hot un-shift for display
        t_low_  = row["T_low"]
 
        annotations.append(dict(
            x=box_x0 - 0.02, y=y_top_box,
            xref="x", yref="y",
            text=f"<b>{t_high_:.1f}°C</b>",
            showarrow=False, font=dict(size=10),
            xanchor="right", yanchor="middle",
        ))
        annotations.append(dict(
            x=box_x0 - 0.02, y=y_bot_box,
            xref="x", yref="y",
            text=f"<b>{t_low_:.1f}°C</b>",
            showarrow=False, font=dict(size=10),
            xanchor="right", yanchor="middle",
        ))
 
        # Interval transfer value inside box
        transfer = vals[i] - vals[i + 1]

        annotations.append(dict(
            x=box_cx, y=(y_top_box + y_bot_box) / 2,
            xref="x", yref="y",
            text=f"<b>ΔH = {transfer:.2f} kW</b>",
            showarrow=False,
            font=dict(size=11, color="red" if is_pinch else "black"),
            xanchor="center", yanchor="middle",
        ))

        if is_pinch:
            pinch_label = " ← Pinch"
            annotations.append(dict(
            x=box_x1, y=(y_bot_box+0.05),
            xref="x", yref="y",
            text=f"<b>{pinch_label}</b>",
            showarrow=False,
            font=dict(size=13, color="red"),
            xanchor="left", yanchor="middle",
        ))
 
        # Arrow to the next box (Between boxes)
        if i < n - 1:
            y_arrow_tail = y_bot_box
            y_arrow_tip  = y_bot_box - gap
 
            transfer = vals[i] - vals[i + 1]
 
            annotations.append(dict(
                x=arrow_x, y=y_arrow_tip,
                ax=arrow_x, ay=y_arrow_tail,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2, arrowsize=1.2, arrowwidth=1.5, arrowcolor="black",
                text="", standoff=0,
            ))
            annotations.append(dict(
                x=label_x, y=(y_arrow_tail + y_arrow_tip) / 2,
                xref="x", yref="y",
                text=f"{val_inside:.2f} kW",
                showarrow=False, font=dict(size=10, color="dimgray"),
                xanchor="left",
            ))
 
    # ── QC arrow exiting at the bottom ───────────────────────────────────────
    qc_val    = vals[-1]
    last_y_bot = y_start - (n - 1) * step - box_h
    qc_y_tail  = last_y_bot - 0.1
    qc_y_tip   = last_y_bot - 0.7
 
    if shifted and qc_val > 1e-6:
        annotations.append(dict(
            x=arrow_x, y=qc_y_tip,
            ax=arrow_x, ay=qc_y_tail,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="steelblue",
            text="", standoff=0,
        ))
        annotations.append(dict(
            x=label_x, y=(qc_y_tail + qc_y_tip) / 2,
            xref="x", yref="y",
            text=f"<b>QC = {qc_val:.2f} kW</b>",
            showarrow=False, font=dict(color="steelblue", size=11),
            xanchor="left",
        ))
    else:
        annotations.append(dict(x=arrow_x, y=qc_y_tip,ax=arrow_x, ay=qc_y_tail,xref="x", yref="y", axref="x", ayref="y",showarrow=True,arrowhead=2, arrowsize=1.2, arrowwidth=1.5, arrowcolor="black",text="", standoff=0))
        annotations.append(dict(x=label_x, y=(qc_y_tail + qc_y_tip) / 2,xref="x", yref="y",text=f"{val_inside:.2f} kW",showarrow=False, font=dict(size=10, color="dimgray"),xanchor="left"))
 
    # ── figure ────────────────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers"))  # dummy trace needed
 
    fig.update_layout(
        title="Shifted Cascade" if shifted else "Unshifted Cascade",
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(visible=False, range=[-0.1, 1.4]),
        yaxis=dict(visible=False, range=[qc_y_tip - 0.3, qh_y_tail + 0.3]),
        template="plotly_white",
        height=max(500, 120 * n),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="white",
    )
    return fig
