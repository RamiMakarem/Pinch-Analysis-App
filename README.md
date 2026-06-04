# Pinch Analysis Tool

A Streamlit-based application for performing pinch analysis and visualizing heat integration opportunities in process systems.

The tool automates the key calculations used in classical pinch analysis and provides interactive visualizations to support process design, energy targeting, and heat exchanger network (HEN) development.

## Live Demo

**Application:** https://pinch-analysis-app.streamlit.app/

> **Note:** The application is hosted on Streamlit Community Cloud. If the app has been inactive for some time, it may enter a sleep state. Simply open the link and allow a short startup period while the application automatically restarts.

---

## Features

### Stream Classification
- Automatic identification of hot and cold streams based on supply and target temperatures.
- Validation and cleaning of user input data.

### Temperature Shifting
- Application of user-defined ΔTmin.
- Automatic generation of shifted temperatures.
- Construction of temperature intervals for pinch calculations.

### Energy Targeting
- Interval heat balance calculations.
- Heat cascade analysis.
- Determination of:
  - Minimum hot utility requirement (QH,min)
  - Minimum cold utility requirement (QC,min)
  - Pinch temperature

### Composite Curves
- Generation of shifted and unshifted composite curves.
- Visualization of heat recovery potential.

### Grand Composite Curve (GCC)
- Automatic GCC generation from cascade results.
- Identification of pinch location and utility requirements.

### Stream Duty Analysis
- Stream heat duty calculations.
- Interval-by-interval duty breakdown.
- Cumulative heat duty tables for both hot and cold streams.

### Heat Cascade Visualization
- Interactive shifted and unshifted cascade diagrams.
- Automatic pinch identification.

### Heat Exchanger Network (HEN) Grid Diagram
- Initial HEN grid representation.
- Visualization of process streams and pinch location.
- Suitable as a starting point for preliminary Heat Exchanger Network synthesis.

---

## Methodology

The application follows the classical pinch analysis workflow:

1. Stream classification
2. Temperature shifting using ΔTmin/2
3. Temperature interval construction
4. Interval heat balance calculations
5. Heat cascade generation
6. Utility target determination
7. Pinch point identification
8. Composite curve generation
9. Grand Composite Curve construction
10. Initial HEN grid generation

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly

---

## Purpose

This project was developed to demonstrate the implementation of pinch analysis methodologies using modern Python-based engineering tools.

The application combines process integration principles with interactive data visualization to provide a practical workflow for:

- Utility targeting
- Pinch point identification
- Composite curve generation
- Grand Composite Curve analysis
- Preliminary Heat Exchanger Network design

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
