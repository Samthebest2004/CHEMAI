import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time  # For animations

# ========== Default Dataset ==========
DEFAULT_DATA = pd.DataFrame({
    "Reactants": ["H2 + O2", "C2H4 + H2", "N2 + H2", "CO + H2O"],
    "Products": ["H2O", "C2H6", "NH3", "CO2"],
    "Concentration (M)": [1.0, 0.8, 0.5, 1.2],
    "Temperature (°C)": [100, 200, 250, 300],
    "Pressure (atm)": [2, 5, 10, 1],
    "Catalyst": ["Pt", "Ni", "Fe", "None"],
    "Cost (INR)": [500, 1000, 2000, 300],
    "Green Chemistry Score": [8.5, 7.2, 9.1, 6.8],
    "Energy Consumption (kWh)": [10, 15, 8, 12],
    "Time (h)": [2, 4, 6, 3]
})

# ========== Load & Validate Dataset ==========
def load_data(file):
    try:
        data = pd.read_csv(file, delimiter='\t')
        required_columns = list(DEFAULT_DATA.columns)  # Ensuring consistency
        if not all(col in data.columns for col in required_columns):
            st.error(f"❌ Missing columns: {', '.join(set(required_columns) - set(data.columns))}")
            return None
        return data
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

# ========== Chemistry-Focused Monte Carlo Simulation (CF-MCS) ==========
def cf_monte_carlo(row, weights, num_simulations=1000):
    adjusted_values = {col: [] for col in row.index if col not in ['Reactants', 'Products', 'Catalyst']}
    
    for _ in range(num_simulations):
        noise = np.random.normal(0, 0.1)  # Gaussian noise

        adjusted_values['Concentration (M)'].append(row['Concentration (M)'] * (1 + weights['concentration'] * noise))
        adjusted_values['Temperature (°C)'].append(row['Temperature (°C)'] * (1 + weights['temperature'] * noise))
        adjusted_values['Pressure (atm)'].append(row['Pressure (atm)'] * (1 + weights['pressure'] * noise))
        adjusted_values['Cost (INR)'].append(max(1, row['Cost (INR)'] * (1 - weights['cost'] * noise)))  
        adjusted_values['Green Chemistry Score'].append(row['Green Chemistry Score'] * (1 + weights['green_chemistry'] * noise))
        adjusted_values['Energy Consumption (kWh)'].append(max(1, row['Energy Consumption (kWh)'] * (1 - weights['energy'] * noise)))  
        adjusted_values['Time (h)'].append(row['Time (h)'] * (1 - weights['time'] * noise))  

    for key in adjusted_values.keys():
        row[key] = np.mean(adjusted_values[key])  # Average over simulations
    return row

# ========== Streamlit App UI ==========
st.title("🧪 AI AUGMENTED Chemistry Analysis: CHEMICAL CALCULATOR :NATIONAL SCIENCE DAY 28FEB 2025 - SAMRAT CHAKRABORTY & DEBDUTTA GHOSH")

# File upload or default dataset
use_default = st.checkbox("📂 Use Default Dataset")
uploaded_file = st.file_uploader("Or Upload a tab-delimited .txt file", type="txt")

if use_default:
    data = DEFAULT_DATA.copy()
elif uploaded_file:
    data = load_data(uploaded_file)
else:
    data = None

if data is not None:
    st.write("📊 **Dataset Preview:**")
    st.dataframe(data)

    # ========== User Selection ==========
    reactant = st.selectbox("🔬 Select Reactant", options=data['Reactants'].unique())
    filtered_data = data[data['Reactants'] == reactant]
    product = st.selectbox("⚗️ Select Product", options=filtered_data['Products'].unique())

    # ========== User-Defined Weights ==========
    st.write("🎛️ **Adjust Priority Sliders:**")
    user_weights = {
        'concentration': st.slider("📈 Concentration", 0.0, 1.0, 0.5),
        'temperature': st.slider("🌡️ Temperature", 0.0, 1.0, 0.5),
        'pressure': st.slider("⏲️ Pressure", 0.0, 1.0, 0.5),
        'cost': st.slider("💰 Cost (Minimize)", 0.0, 1.0, 0.5),
        'green_chemistry': st.slider("♻️ Green Chemistry Score", 0.0, 1.0, 0.5),
        'energy': st.slider("⚡ Energy Consumption", 0.0, 1.0, 0.5),
        'time': st.slider("⏳ Time (Minimize)", 0.0, 1.0, 0.5)
    }

    # ========== Apply CF-MCS ==========
    filtered_data = filtered_data[filtered_data['Products'] == product]
    if not filtered_data.empty:
        st.write("🔄 **Optimizing with CF-MCS...**")
        time.sleep(1)  # Simulate processing delay
        optimized_data = filtered_data.apply(cf_monte_carlo, axis=1, weights=user_weights)

        # ========== Chemistry-Optimized Pareto Approximation (CO-PA) ==========
        optimized_data['Rank'] = (
            optimized_data['Concentration (M)'] * user_weights['concentration'] +
            optimized_data['Temperature (°C)'] * user_weights['temperature'] +
            optimized_data['Pressure (atm)'] * user_weights['pressure'] +
            (10 - optimized_data['Cost (INR)']) * user_weights['cost'] +
            optimized_data['Green Chemistry Score'] * user_weights['green_chemistry'] +
            (10 - optimized_data['Energy Consumption (kWh)']) * user_weights['energy'] +
            (10 - optimized_data['Time (h)']) * user_weights['time']
        )
        ranked_data = optimized_data.sort_values(by='Rank', ascending=False)

        st.write("🏆 **Top Ranked Reactions:**")
        st.dataframe(ranked_data)

        # ========== Visualization ==========
        ranked_data['Reaction'] = ranked_data['Reactants'] + " → " + ranked_data['Products']
        fig = px.bar(ranked_data, x='Reaction', y='Green Chemistry Score', 
                     title='♻️ Green Chemistry Score of Ranked Reactions',
                     labels={'Reaction': 'Reaction', 'Green Chemistry Score': 'Score'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)

        # ========== Emoji Animation ==========
        st.write("🎉 **Optimization Complete!** 🔔")
        time.sleep(1)
        st.write("🔔🔔🔔")

else:
    st.info("📂 Please upload a file or use the default dataset.")

