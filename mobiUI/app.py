from io import BytesIO
import pandas as pd
import pydeck as pdk
import streamlit as st
import json
import requests
from PIL import Image
import io
import os

HOST_XAI = os.getenv("HOST_XAI"); HOST_AI = os.getenv("HOST_AI")
PORT_XAI = os.getenv("PORT_XAI"); PORT_AI = os.getenv("PORT_AI")
XAI_URL = f"{HOST_XAI}:{PORT_XAI}"  # HOST+"/ai"
AI_URL = f"{HOST_AI}:{PORT_AI}"  # HOST+"/xai"


def is_json(jdata):
    try: jdata = json.loads(jdata); return [True, jdata]
    except ValueError: return [False]


def plot_map(actual_points, predicted_points, lat, lon):
    tooltip = {  # Define tooltips
        "html": "<b>Index:</b> {index}<br><b>Type:</b> {pred_text}<br><b>Latitude:</b> {lat}"
                "<br><b>Longitude:</b> {lon}",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
    base_layers = [  # Base layers for actual and predicted data points
        pdk.Layer(
            'ScatterplotLayer', data=pd.read_json(actual_points, orient='split'),  # Actual points
            get_position='[lon, lat]', get_color='[250, 255, 0, 160]',  # Green color for actual points
            get_radius=100, pickable=True, auto_highlight=True,
        ),
        pdk.Layer(
            'ScatterplotLayer', data=pd.read_json(predicted_points, orient='split'),  # Predicted points
            get_position='[lon, lat]', get_color='[0, 0, 250, 160]',  # Red color for predicted points
            get_radius=100, pickable=True, auto_highlight=True,
        )
    ]
    # Visualization with PyDeck
    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=11, pitch=50)
    col1.pydeck_chart(pdk.Deck(  # Render the map with the layers and tooltip
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=view_state, layers=base_layers, tooltip=tooltip
    ))


st.set_page_config(page_title='XAI demonstration', page_icon=":bar_chart:", layout="wide")
st.title('T5.1 & T5.3 | MobiBench → FLP → XAI by UPRC')

with st.container():
    col1_header, col2_header, col3_header = st.columns([3, 5, 1])
    with col1_header: st.markdown("## Future Location Prediction Model (FLP) ")
    with col2_header: st.markdown("## eXplainable AI (XAI)")
    col1, col2, col3 = st.columns(3)


# # Streamlit app interface
st.sidebar.image("MobiSpaces_logo.jpg")
st.sidebar.header("Please choose a model")
model_flp = st.sidebar.selectbox('FLP model', ['Short-Term | 20m Horizon', 'Mid-Term | 30m Horizon'])
st.sidebar.write('**MobiBench**')
try:
    resp = requests.post(f"{AI_URL}/spider_chart")  # TODO
    if resp.ok: st.sidebar.image(BytesIO(resp.content))
    else: st.sidebar.write(resp.text)
except Exception: st.sidebar.write("There was a server error.")


##############################################
if model_flp == "Mid-Term | 30m Horizon": model_no = "b"
else: model_no = "a"
tid_lst = []; selected_tid = 0; max_value = 0; unique_lh_values = []; map_data = {}; data = {}; selected_lh = 0

try:
    # Dropdown for selecting Trip ID
    resp = requests.get(f"{AI_URL}/trips?model={model_no}")  # TODO
    if resp.ok:
        data = is_json(resp.content)
        if data[0]:
            tid_lst = data[1]["trip_ids"]; max_value = data[1]["max_value"]
        else: raise Exception()
        selected_tid = col1.selectbox('Trip ID', options=tid_lst, index=0)
        # Creating a slider for lh values based on their index positions
        lh_index = col1.slider('Prediction Horizon Index (mins)', min_value=5, max_value=max_value, value=5, step=5)
    else: raise Exception()

    # Call the plotting function with both selections
    resp = requests.get(f"{AI_URL}/map_data?model={model_no}&sid={selected_tid}&slh={int(lh_index)}")  # TODO
    if resp.ok:
        map_data = is_json(resp.content)
        if map_data[0]: map_data = map_data[1]
        else: raise Exception
        plot_map(map_data["actual_points"], map_data["predicted_points"], map_data["lat"], map_data["lon"])
        col1.write("Map Legend:")
        col1.markdown("""
        - **Yellow**: Actual data points
        - **Blue**: Predicted data points
        """)
    else: raise Exception()

except Exception: col1.write("There was a server error.")


##############################################
# Generate and display the feature importance plot
col2.subheader('Feature Importance')
col2.write('This visualization ranks input features by their importance in influencing the model\'s predictions, '
           'with higher scores indicating greater impact.')

with st.spinner('Generating feature importance...'):
    resp = requests.get(f"{XAI_URL}/feature_importance?model={model_no}")  # TODO
    if resp.ok: col2.image(Image.open(io.BytesIO(resp.content)))
    else: col2.write("There was a server error.")


###################################################
col3.subheader('Predicted point Explanation')
col3.write('This LIME plot shows the impact of individual features on the model\'s prediction for a specific '
           'instance. Features on the X-axis; their predictive influence on the Y-axis. Positive values indicate '
           'features supporting the prediction, while negative values detract from it.')

try:
    values_list = {}
    resp = requests.get(f"{XAI_URL}/index_instances?model={model_no}&sid={selected_tid}")  # TODO
    if resp.ok:
        data = is_json(resp.content)
        if data[0]: values_list = data[1]
    else: raise Exception()
    instance_index = col3.number_input('Enter the index of the instance you want to explain:',
                                       min_value=int(values_list["min_value"]), max_value=int(values_list["max_value"]),
                                       value=int(values_list["min_value"]), step=1)

    explanation = ""
    with st.spinner('Generating explanation...'):
        resp = requests.get(f"{XAI_URL}/explain?model={model_no}&sid={selected_tid}&index={instance_index}")  # TODO
        if resp.ok:
            explanation = resp.text
            resp2 = requests.post(f"{XAI_URL}/explanation_figure", data=resp.text)  # TODO
            if resp2.ok: col3.image(Image.open(io.BytesIO(resp2.content)))
            else: raise Exception()
        else: raise Exception()

    # Get available LLM providers
    try:
        resp = requests.get(f"{XAI_URL}/llm_providers")
        if resp.ok:
            llm_data = resp.json()
            providers = llm_data["providers"]
            default_provider = llm_data["default_provider"]
            default_model = llm_data["default_model"]
            default_temperature = llm_data["default_temperature"]
        else:
            providers = {"openai": {"available_models": ["gpt-4-turbo-preview"], "api_key_valid": False}}
            default_provider = "openai"
            default_model = "gpt-4-turbo-preview"
            default_temperature = 0.0
    except Exception:
        providers = {"openai": {"available_models": ["gpt-4-turbo-preview"], "api_key_valid": False}}
        default_provider = "openai"
        default_model = "gpt-4-turbo-preview"
        default_temperature = 0.0

    # Creating an expander for LLM explanations
    with col3.expander("Click here to expand and see more details from LLM"):
        # Provider selection
        available_providers = [p for p, info in providers.items() if info.get("api_key_valid", False)]
        if not available_providers:
            st.warning("No LLM providers with valid API keys found. Please configure API keys.")
            available_providers = list(providers.keys())
        
        selected_provider = st.selectbox(
            "Select LLM Provider:", 
            options=available_providers,
            index=available_providers.index(default_provider) if default_provider in available_providers else 0
        )
        
        # Model selection
        provider_info = providers.get(selected_provider, {})
        available_models = provider_info.get("available_models", [])
        default_model_for_provider = provider_info.get("default_model", available_models[0] if available_models else "")
        
        selected_model = st.selectbox(
            "Select Model:", 
            options=available_models,
            index=available_models.index(default_model_for_provider) if default_model_for_provider in available_models else 0
        )
        
        # Temperature slider
        temperature = st.slider("Temperature:", min_value=0.0, max_value=2.0, value=float(default_temperature), step=0.1)
        
        # Max tokens slider
        max_tokens = st.slider("Max Tokens:", min_value=100, max_value=4000, value=1000, step=100)
        
        if st.button("Generate LLM Explanation"):
            payload = json.dumps({
                "explanation": explanation,
                "provider": selected_provider,
                "model": selected_model,
                "temperature": temperature,
                "max_tokens": max_tokens
            })
            resp = requests.post(f"{XAI_URL}/chat_response", data=payload)
            if resp.ok:
                st.write(resp.text)
            else:
                st.error(f"Error: {resp.text}")
except Exception: col3.write("There was a server error.")
