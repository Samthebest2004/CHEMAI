import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO

# Default dataset as a multi-line string
default_data = """Reactants	Products	Concentration (M)	Temperature (°C)	Pressure (atm)	Catalyst	Cost (INR)	Green Chemistry Score	Energy Consumption (kWh)	Time (h)
benzene	phenol	2	250	10	FeCl3	7400	7	150	5
benzene	phenol	1.5	220	8	V2O5	5920	8	120	4
benzene	aniline	1.8	180	7	Pd/C	6660	8	110	3
ethanol	acetic acid	1	150	5	None	3700	9	80	2
ethanol	acetaldehyde	0.5	140	4	None	2960	9	70	1.5
methanol	formaldehyde	0.8	120	3	None	4440	8	65	2
aniline	azobenzene	1.2	160	6	NaNO2	8140	6	95	3.5
toluene	benzoic acid	1	180	7	None	5180	7	100	3
cyclohexanol	cyclohexanone	0.9	170	5	None	5920	8	85	2.5
acetic acid	ethyl acetate	1	130	4	None	2220	9	60	3
glucose	ethanol	0.8	35	1	Yeast	1480	10	20	48
acetylene	benzene	1.5	200	5	Ni(CO)4	9620	5	130	4
acetylene	vinyl chloride	1	150	4	CuCl	7770	6	105	3
propene	propylene oxide	0.6	120	3	Ag2O	5550	8	75	2
formaldehyde	urea-formaldehyde resin	0.5	90	1	NaOH	3700	5	50	6
acetone	isopropanol	1	140	3	Ni	4440	9	60	2
chloroethane	ethanol	1	130	4	None	5180	8	70	2.5
styrene	polystyrene	0.5	80	1	Peroxide	4070	7	55	12
butane	butanol	0.7	150	5	CuO	6660	7	90	3
phenol	aniline	1.2	160	6	Pd/C	7030	8	95	3.5
hexane	hexanol	0.9	170	5	CuO	6660	7	90	3
hexanol	hexanoic acid	1.1	180	6	None	6290	8	85	3.5
benzaldehyde	benzoic acid	0.7	150	4	None	5550	9	75	2
formic acid	carbon dioxide	0.5	120	3	None	4440	8	60	1
propene	acrylic acid	1	160	5	Mo-V Catalyst	7400	7	100	4
acetone	mesitylene	1.2	170	5	None	8140	6	110	4
benzyl chloride	benzyl alcohol	1	140	4	None	5180	8	70	2.5
glucose	sorbitol	0.8	130	3	Ni	4440	9	60	2"""

# Function to load and validate dataset
def load_data(file):
    try:
        data = pd.read_csv(file, delimiter='\t')
        required_columns = [
            'Reactants', 'Products', 'Concentration (M)', 'Temperature (°C)', 'Pressure (atm)', 
            'Catalyst', 'Cost (INR)', 'Green Chemistry Score', 'Energy Consumption (kWh)', 'Time (h)'
        ]
        if not all(column in data.columns for column in required_columns):
            missing_cols = [col for col in required_columns if col not in data.columns]
            st.error(f"Missing columns in dataset: {', '.join(missing_cols)}")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Monte Carlo Simulation for parameter adjustment
def monte_carlo_adjustment(row, weights, num_simulations=1000):
    adjusted_values = {col: [] for col in row.index if col not in ['Reactants', 'Products', 'Catalyst']}
    
    # We'll include the parameters that we want to adjust
    for _ in range(num_simulations):
        noise = np.random.normal(loc=0, scale=0.1)
        adjusted_values.setdefault('Concentration (M)', []).append(row['Concentration (M)'] * (1 + weights['concentration'] * noise))
        adjusted_values.setdefault('Temperature (°C)', []).append(row['Temperature (°C)'] * (1 + weights['temperature'] * noise))
        adjusted_values.setdefault('Pressure (atm)', []).append(row['Pressure (atm)'] * (1 + weights['pressure'] * noise))
        adjusted_values.setdefault('Cost (INR)', []).append(max(1, row['Cost (INR)'] * (1 - weights['cost'] * noise)))
        adjusted_values.setdefault('Green Chemistry Score', []).append(row['Green Chemistry Score'] * (1 + weights['green_chemistry'] * noise))
        adjusted_values.setdefault('Energy Consumption (kWh)', []).append(max(1, row['Energy Consumption (kWh)'] * (1 - weights['energy'] * noise)))
        adjusted_values.setdefault('Time (h)', []).append(row['Time (h)'] * (1 - weights['time'] * noise))
    
    # Take the mean of the simulated values and update the row accordingly.
    for key, values in adjusted_values.items():
        row[key] = np.mean(values)
    return row

# Streamlit App Title
st.title("🧪 AI AUGMENTED CHEMICAL ANALYSIS: CHEMISTRY CALCULATOR - SAMRAT CHAKRABORTY")

# Option selection: Default dataset or upload your own
data_source = st.radio("Choose data source:", ("Default Dataset", "Upload your Dataset"))

if data_source == "Default Dataset":
    # Use the default dataset by reading the multi-line string
    file = StringIO(default_data)
    data = load_data(file)
else:
    uploaded_file = st.file_uploader("📂 Upload a tab-delimited .txt file", type="txt")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
    else:
        st.info("📂 Please upload a tab-delimited .txt file.")
        data = None

if data is not None:
    st.write("📊 **Dataset Preview:**")
    st.dataframe(data.head())

    # User selects Reactant and Product
    reactant = st.selectbox("🔬 Select Reactant", options=data['Reactants'].unique())
    filtered_data = data[data['Reactants'] == reactant]
    product = st.selectbox("⚗️ Select Product", options=filtered_data['Products'].unique())

    # User interaction: Sliders for importance weights
    st.write("🎛️ **Adjust the sliders to set priorities:**")
    user_weights = {
        'concentration': st.slider("📈 Concentration (M)", 0.0, 1.0, 0.5),
        'temperature': st.slider("🌡️ Temperature (°C)", 0.0, 1.0, 0.5),
        'pressure': st.slider("⏲️ Pressure (atm)", 0.0, 1.0, 0.5),
        'catalyst': st.slider("🧪 Catalyst (Importance)", 0.0, 1.0, 0.5),
        'cost': st.slider("💰 Cost (Minimize)", 0.0, 1.0, 0.5),
        'green_chemistry': st.slider("♻️ Green Chemistry Score", 0.0, 1.0, 0.5),
        'energy': st.slider("⚡ Energy Consumption (Minimize)", 0.0, 1.0, 0.5),
        'time': st.slider("⏳ Time (Minimize)", 0.0, 1.0, 0.5)
    }

    # Filter data for the selected reaction
    filtered_data = filtered_data[filtered_data['Products'] == product]
    if not filtered_data.empty:
        # Apply Monte Carlo adjustment for each row
        adjusted_data = filtered_data.apply(monte_carlo_adjustment, axis=1, weights=user_weights)

        # Calculate rank based on user preferences
        adjusted_data['Rank'] = (
            adjusted_data['Concentration (M)'] * user_weights['concentration'] +
            adjusted_data['Temperature (°C)'] * user_weights['temperature'] +
            adjusted_data['Pressure (atm)'] * user_weights['pressure'] +
            adjusted_data['Catalyst'].apply(lambda x: user_weights['catalyst'] if pd.notna(x) and x != "None" else 0) +
            (10 - adjusted_data['Cost (INR)']) * user_weights['cost'] +
            adjusted_data['Green Chemistry Score'] * user_weights['green_chemistry'] +
            (10 - adjusted_data['Energy Consumption (kWh)']) * user_weights['energy'] +
            (10 - adjusted_data['Time (h)']) * user_weights['time']
        )
        ranked_data = adjusted_data.sort_values(by='Rank', ascending=False)

        st.write("🏆 **Top Ranked Reactions:**")
        st.dataframe(ranked_data)

        # Convert 'Reactants' and 'Products' to strings before concatenating
        ranked_data['Reaction'] = ranked_data['Reactants'].astype(str) + " → " + ranked_data['Products'].astype(str)
        fig = px.bar(ranked_data, x='Reaction', y='Green Chemistry Score', 
                     title='♻️ Green Chemistry Score of Ranked Reactions',
                     labels={'Reaction': 'Reaction', 'Green Chemistry Score': 'Green Chemistry Score'})
        fig.update_layout(xaxis_title='Reaction', yaxis_title='Green Chemistry Score', xaxis_tickangle=-45)
        st.plotly_chart(fig)
