import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from core import (
    classify_streams,
    adjust_temperatures,
    calculate_delta_h,
    cascade,
    stream_energy_table,
    build_stream_interval_tables,
    build_composite_curves,
    build_gcc_curve,
    build_hen_grid_data,
    plot_cascade_diagram
)

# ─────────────────────────────────────────────
# PLOT FUNCTIONS
# ─────────────────────────────────────────────

def plot_composite_curves(hot_segments, cold_segments, title="Composite Curves"):
    """
    Hot and cold composite curves on the same H-T axis.
    Hot starts at H=0, cold is shifted right by qh (applied in build_composite_curves).
    """
    fig = go.Figure()

    for i, seg in enumerate(hot_segments):
        H = [p[0] for p in seg]
        T = [p[1] for p in seg]
        fig.add_trace(go.Scatter(
            x=H, y=T,
            mode="lines+markers",
            line=dict(color="firebrick", width=3),
            marker=dict(size=6, color="firebrick"),
            name="Hot Composite" if i == 0 else None,
            legendgroup="hot", showlegend=(i == 0),
            hovertemplate="H = %{x:.2f} kW<br>T = %{y:.1f} °C<extra>Hot</extra>",
        ))

    for i, seg in enumerate(cold_segments):
        H = [p[0] for p in seg]
        T = [p[1] for p in seg]
        fig.add_trace(go.Scatter(
            x=H, y=T,
            mode="lines+markers",
            line=dict(color="steelblue", width=3),
            marker=dict(size=6, color="steelblue"),
            name="Cold Composite" if i == 0 else None,
            legendgroup="cold", showlegend=(i == 0),
            hovertemplate="H = %{x:.2f} kW<br>T = %{y:.1f} °C<extra>Cold</extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Enthalpy H (kW)",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        height=460,
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def plot_gcc(cascade_df, delta_tmin):
    H_adj, T_star = build_gcc_curve(cascade_df, delta_tmin)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=H_adj, y=T_star,
        mode="lines+markers",
        name="GCC",
        line=dict(color="darkorange", width=3),
        marker=dict(size=6),
    ))

    # Pinch nose — point where H = 0
    for h, t in zip(H_adj, T_star):
        if abs(h) < 1e-6:
            fig.add_trace(go.Scatter(
                x=[h], y=[t],
                mode="markers",
                marker=dict(color="black", size=10, symbol="diamond"),
                name=f"Pinch ({t:.1f} °C*)",
                showlegend=True,
            ))

    fig.update_layout(
        title="Grand Composite Curve",
        xaxis_title="Net Heat Flow (kW)",
        yaxis_title="Shifted Temperature T* (°C)",
        template="plotly_white",
    )
    return fig


def plot_stream_heat(df):
    df = df.copy()
    df["Q"] = df["CP"] * abs(df["Tin"] - df["Tout"])
    fig = px.bar(
        df, x="Stream ID", y="Q", color="Type",
        color_discrete_map={"Hot": "firebrick", "Cold": "steelblue"},
        title="Stream Heat Duties",
        labels={"Q": "Heat Duty (kW)"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_hen_grid(hot_streams, cold_streams, pinch, pinch_cold, delta_tmin):
    """
    HEN grid diagram.
    - Filled circles at BOTH supply and target ends with stream ID inside
    - Temperatures above the line
    - Single continuous pinch line spanning full height, both temps labeled
    """

    fig = go.Figure()

    row_gap      = 1.8
    hot_y_base   =  1.2
    cold_y_base  = -1.2
    label_offset =  0.32
    circ_size    =  30

    n_hot  = len(hot_streams)
    n_cold = len(cold_streams)

    all_temps = (
        [s["Tin"]  for s in hot_streams + cold_streams]
        + [s["Tout"] for s in hot_streams + cold_streams]
        + [pinch, pinch_cold]
    )
    t_min = min(all_temps) - 20
    t_max = max(all_temps) + 20

    y_top = hot_y_base  + (n_hot  - 1) * row_gap + 1.4
    y_bot = cold_y_base - (n_cold - 1) * row_gap - 1.4

    def _draw_stream(s, y, color):
        t_sup = s["Tin"]
        t_tgt = s["Tout"]
        sid   = s["id"]

        # stream line
        fig.add_trace(go.Scatter(
            x=[t_sup, t_tgt], y=[y, y],
            mode="lines",
            line=dict(color=color, width=3),
            showlegend=False, hoverinfo="skip",
        ))

        # supply circle + stream ID
        fig.add_trace(go.Scatter(
            x=[t_sup], y=[y],
            mode="markers+text",
            marker=dict(symbol="circle", size=circ_size, color=color,
                        line=dict(color="white", width=1.5)),
            text=[sid], textposition="middle center",
            textfont=dict(color="white", size=10, family="Arial Black"),
            showlegend=False,
            hovertemplate=(
                f"<b>{sid}</b><br>{t_sup}°C → {t_tgt}°C<br>"
                f"CP={s['CP']} kW/°C | Q={s['Q']:.1f} kW<extra></extra>"
            ),
        ))

        # target circle + stream ID
        fig.add_trace(go.Scatter(
            x=[t_tgt], y=[y],
            mode="markers+text",
            marker=dict(symbol="circle", size=circ_size, color=color,
                        line=dict(color="white", width=1.5)),
            text=[sid], textposition="middle center",
            textfont=dict(color="white", size=10, family="Arial Black"),
            showlegend=False,
            hovertemplate=(
                f"<b>{sid}</b> target: {t_tgt}°C<extra></extra>"
            ),
        ))

        # temperature labels above line
        for t in [t_sup, t_tgt]:
            fig.add_annotation(
                x=t, y=y + label_offset,
                text=f"<b>{t}°C</b>",
                showarrow=False,
                font=dict(size=11, color=color),
                xanchor="center", yanchor="bottom",
            )

    for i, s in enumerate(hot_streams):
        _draw_stream(s, hot_y_base  + i * row_gap,  "firebrick")

    for i, s in enumerate(cold_streams):
        _draw_stream(s, cold_y_base - i * row_gap, "steelblue")

    # ── single pinch line, full height ───────────────────────────────────────
    fig.add_shape(
        type="line",
        x0=pinch, x1=pinch,
        y0=y_bot, y1=y_top,
        line=dict(color="black", width=2.5, dash="dash"),
        layer="below",
    )
    fig.add_annotation(
        x=pinch, y=y_top,
        text=(
            f"<b>Pinch</b><br>"
            f"T* = {pinch:.1f}°C<br>"
            f"Hot side: {pinch + delta_tmin/2:.1f}°C<br>"
            f"Cold side: {pinch - delta_tmin/2:.1f}°C"
        ),
        showarrow=False,
        font=dict(size=11, color="black"),
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="black", borderwidth=1, borderpad=4,
        xanchor="center", yanchor="bottom",
    )

    # ── hot / cold divider ────────────────────────────────────────────────────
    fig.add_shape(type="line", x0=t_min, x1=t_max, y0=0, y1=0,
                  line=dict(color="lightgray", width=1))
    fig.add_annotation(x=t_min+3, y= 0.38, text="HOT STREAMS", showarrow=False,
                       font=dict(size=10, color="firebrick"), xanchor="left")
    fig.add_annotation(x=t_min+3, y=-0.38, text="COLD STREAMS", showarrow=False,
                       font=dict(size=10, color="steelblue"), xanchor="left")

    # ── above / below pinch labels ────────────────────────────────────────────
    fig.add_annotation(x=(pinch+t_max)/2, y=y_top, text="Above Pinch",
                       showarrow=False, font=dict(size=18, color="gray"))
    fig.add_annotation(x=(t_min+pinch)/2, y=y_top, text="Below Pinch",
                       showarrow=False, font=dict(size=18, color="gray"))

    fig.update_layout(
        title="HEN Grid Diagram (Initial — before heat exchangers)",
        xaxis=dict(title="Temperature (°C)", range=[t_min, t_max],
                   showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
        yaxis=dict(visible=False, range=[y_bot - 0.5, y_top + 1.8]),
        template="plotly_white",
        height=max(440, 150 * (n_hot + n_cold)),
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor="white",
    )
    return fig


# ─────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────

st.set_page_config(page_title="Pinch Analysis Tool", layout="wide")
st.title("Pinch Analysis Tool")

# ── Stream input ──────────────────────────────
st.subheader("Stream Data")

df_input = pd.DataFrame({
    "Stream ID": ["H1", "H2", "C1", "C2"],
    "Tin":       [180,  150,  30,   60],
    "Tout":      [60,   30,   120,  180],
    "CP":        [2.5,  4.0,  3.0,  2.0],
})

edited_df = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)

# ── Parameters ───────────────────────────────
col_a, col_b = st.columns(2)
with col_a:
    delta_tmin = st.number_input("ΔTmin (°C)", value=10.0, min_value=1.0, step=1.0)

run_button = st.button("▶ Run Analysis", type="primary")

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────

if run_button:

    # 1. Classify
    df = classify_streams(edited_df)

    with st.expander("Classified Streams", expanded=False):
        st.dataframe(df, use_container_width=True)

    # 2. Stream heat duties chart
    st.subheader("Stream Heat Duties")
    st.plotly_chart(plot_stream_heat(df), use_container_width=True)

    energy_df = stream_energy_table(df)
    with st.expander("Stream Energy Table", expanded=False):
        st.dataframe(energy_df, use_container_width=True)

    # 3. Adjust temperatures + intervals
    df_adj, intervals = adjust_temperatures(df, delta_tmin)

    with st.expander("Adjusted Temperatures", expanded=False):
        st.dataframe(df_adj, use_container_width=True)

    # 4. ΔH table
    delta_h_df = calculate_delta_h(df_adj, intervals)

    with st.expander("ΔH Interval Table", expanded=False):
        st.dataframe(delta_h_df, use_container_width=True)

    # 4b. Stream interval duty tables
    st.subheader("Stream Interval Duty Tables")
    hot_interval_df, cold_interval_df = build_stream_interval_tables(df_adj, intervals)

    st.markdown("**Hot Streams — Duty per Interval (kW)**")
    st.dataframe(hot_interval_df, use_container_width=True, hide_index=True)

    st.markdown("**Cold Streams — Duty per Interval (kW)**")
    st.dataframe(cold_interval_df, use_container_width=True, hide_index=True)

    # 5. Cascade
    cascade_df, qh, qc, pinch_temp, pinch_interval = cascade(delta_h_df)

    with st.expander("Cascade Table", expanded=False):
        st.dataframe(cascade_df, use_container_width=True)
    
    # 5b. Cascade diagrams
    st.subheader("Cascade Diagrams")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Unshifted Cascade")
        unshifted_fig = plot_cascade_diagram(cascade_df,shifted=False)
        st.plotly_chart(unshifted_fig, use_container_width=True)

    with col2:
        st.markdown("### Shifted Cascade")
        shifted_fig = plot_cascade_diagram(cascade_df,shifted=True)
        st.plotly_chart(shifted_fig, use_container_width=True)

    # 6. Summary metrics
    st.subheader("Heat Integration Summary")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Min Hot Utility QH,min", f"{qh:.2f} kW")
    with m2:
        st.metric("Min Cold Utility QC,min", f"{qc:.2f} kW")
    with m3:
        if pinch_temp is not None:
            st.metric("Pinch Temperature", f"{pinch_temp:.2f} °C")
        else:
            st.metric("Pinch Temperature", "Threshold problem")

    # 7. Composite curves
    st.subheader("Composite Curves")
    
    unshifted_hot_segments, unshifted_cold_segments = build_composite_curves(df_adj, intervals, 0, qc)
    st.plotly_chart(plot_composite_curves(unshifted_hot_segments, unshifted_cold_segments, "Composite Curves (Shifted Temperatures)"), use_container_width=True)
    
    hot_segments, cold_segments = build_composite_curves(df_adj, intervals, delta_tmin, qc)
    st.plotly_chart(plot_composite_curves(hot_segments, cold_segments, "Composite Curves (Unshifted Temperatures)"), use_container_width=True)
    

    # 8. Grand Composite Curve
    st.subheader("Grand Composite Curve")
    st.plotly_chart(plot_gcc(cascade_df, delta_tmin), use_container_width=True)

    # 9. HEN Grid (initial)
    st.subheader("HEN Grid Diagram")
    if pinch_temp is not None:
        hot_streams, cold_streams, pinch, pinch_cold = build_hen_grid_data(df, pinch_temp, delta_tmin)
        st.plotly_chart(
            plot_hen_grid(hot_streams, cold_streams, pinch, pinch_cold, delta_tmin),use_container_width=True)
    else:
        st.info("No pinch point found (threshold problem) — HEN grid not applicable.")

else:
    st.info("Enter stream data and click **▶ Run Analysis** to begin.")