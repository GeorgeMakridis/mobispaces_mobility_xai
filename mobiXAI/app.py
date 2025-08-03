import io
import pandas as pd
from lime import lime_tabular
import numpy as np
import xgboost as xgb
from flask import Flask, request, Response
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import json
import requests
import os
from llm_providers.factory import LLMFactory

# Environment variables for LLM providers
HOST_CONNECTOR = os.getenv("HOST_CONNECTOR")
PORT_CONNECTOR = os.getenv("PORT_CONNECTOR")
CONNECTOR_URL = f"{HOST_CONNECTOR}:{PORT_CONNECTOR}"

# LLM API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# Default LLM settings
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4-turbo-preview")
DEFAULT_LLM_TEMPERATURE = float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.0"))


def run_app():
    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False
    app.config['JSON_SORT_KEYS'] = False

    col_lists = {
        "b": ['oid', 'lon', 'lat', 'speed', 'bearing', 'hour', 'weekday', 'ship_type',
              'origin', 'start_hour', 'start_weekday', 'extrap_lon', 'extrap_lat', 'lh'],
        "a": ['oid', 'lon', 'lat', 'speed', 'bearing', 'hour', 'weekday', 'ship_type',
              'origin', 'start_hour', 'start_weekday', 'extrap_lon', 'extrap_lat', 'lh'],
    }

    def is_json(data):
        try: data = json.loads(data); return [True, data]
        except ValueError: return [False]

    @app.route('/explain', methods=['GET'])
    def explain():
        model_no = "a"
        if request.args:
            arguments = request.args.to_dict()
            if "model" in arguments:
                model_no = arguments["model"]
                if model_no not in ["a", "b"]: model_no = "a"
        sid = None; index_input = None
        if request.args:
            arguments = request.args.to_dict()
            if "sid" not in arguments: return "Missing arguments.", 400
            if "index" not in arguments: return "Missing arguments.", 400
            sid = int(arguments["sid"]); index_input = int(arguments["index"])
        if sid is None or index_input is None: return "Missing arguments.", 400
        # Load XGBoost model (adjust the path as needed)
        bst_gpu = xgb.XGBRegressor()
        if model_no == "b": bst_gpu.load_model('/mobi_data/model_b_model.pkl')
        else: bst_gpu.load_model('/mobi_data/model_a_model.ubj')
        col_list_case = col_lists[model_no]
        return explain_with_lime(bst_gpu, sid, index_input, col_list_case, model_no=model_no)

    @app.route('/explanation_figure', methods=['POST'])
    def explanation_figure():
        data = is_json(request.data); as_image = 1
        if data[0] is False: return "Missing arguments.", 400
        else: explanation = data[1]
        if request.args:
            arguments = request.args.to_dict()
            if "data" in arguments: as_image = 0
        features, importances = zip(*explanation)  # Unpack feature names and importances
        # Generate colors based on the sign of the importance values
        colors = ['green' if importance > 0 else 'red' for importance in importances]
        if as_image == 0:
            data = {"title": "Filtered Feature Importance with Color Coding", "values_label": "Importance",
                    "data": [
                        {"label": f"{features[i]}", "value": f"{importances[i]}", "color": colors[i]}
                        for i in range(0, len(features))
                    ]}
            return data

        fig, ax = plt.subplots()
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_xlabel('Importance')
        ax.set_title('Filtered Feature Importance with Color Coding')
        plt.tight_layout()

        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')

    @app.route('/feature_importance', methods=['GET'])
    def feature_importance():
        model_no = "a"; as_image = 1
        if request.args:
            arguments = request.args.to_dict()
            if "model" in arguments:
                model_no = arguments["model"]
                if model_no not in ["a", "b"]: model_no = "a"
            if "data" in arguments: as_image = 0

        # Load XGBoost model (adjust the path as needed)
        bst_gpu = xgb.XGBRegressor()
        if model_no == "b": bst_gpu.load_model('/mobi_data/model_b_model.pkl')
        else: bst_gpu.load_model('/mobi_data/model_a_model.ubj')
        col_list_case = col_lists[model_no]
        resp = plot_feature_importance(bst_gpu, col_list_case, as_image)
        if as_image == 0: return json.dumps(str(resp)), 200
        output = io.BytesIO()
        FigureCanvas(resp).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')

    def explain_with_lime(model, selected_tid, index_input, feature_names, num_features=10, model_no=""):
        """
        Generates LIME explanations for a specific data instance.

        :param model: The trained model to explain.
        :param selected_tid: .
        :param index_input: .
        :param feature_names: List of feature names.
        :param num_features: Number of features to include in the explanation.
        :param model_no: specifies the model that should be loaded.
        :return: LIME explanation object.
        """
        # Now use 'filtered_samples' and 'filtered_feature_names' for generating LIME explanations
        df = pd.read_pickle(f'mobi_data/model_{model_no}_data.pkl')
        # # connector
        # data_url = f"{CONNECTOR_URL}/api/connector/csv/{model_no}"
        # response = requests.get(data_url)
        # response.raise_for_status()
        # csv_buffer = io.StringIO(response.content.decode("utf-8"))
        # df = pd.read_csv(csv_buffer)

        df.dropna(inplace=True)
        if all(column in df.columns for column in feature_names): X = df[feature_names]
        else: return "The DataFrame does not contain all the required features for prediction.", 400

        filtered_df = df[df['_tid'] == selected_tid]
        # Assuming 'features' contains the list of feature column names used by your model
        X_filtered = filtered_df[feature_names]
        selected_row = X_filtered.iloc[[index_input]]  # Example: select the row to explain

        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X.values,  # X_train should be your training data as a numpy array
            feature_names=feature_names,
            class_names=['lat', 'lon'],  # Adjust based on your use case
            mode='regression'  # Use 'classification' for classification models
        )

        # Model prediction function
        # predict_fn = lambda x: model.predict(x).reshape(-1, 1)

        # Generate explanation for the data instance
        # Corrected to directly pass data_instance.values[0] without naming the argument
        # Assuming data_instance is a Pandas Series
        explanation = explainer.explain_instance(
            data_row=selected_row.values[0],  # Reshape to 2D
            predict_fn=model.predict, num_features=num_features
        )

        exclude_features = ["lat", "lon", "extrap_lon", "extrap_lat", "lh"]

        # Get the list of features and their importances from the explanation
        explanation_list = explanation.as_list()

        # Filter out the features where the base name is in exclude_features
        filtered_explanation = [
            (feature, importance) for feature, importance in explanation_list
            if not any(exclude_feature in feature for exclude_feature in exclude_features)
        ]
        return filtered_explanation, 200

    def plot_feature_importance(model, col_list, as_image=1):
        importances = model.feature_importances_  # Get feature importances
        # Assuming you have a list of feature names
        feature_names = col_list  # Adjust as per your features

        # Combine the importances and feature names, and filter
        importance_dict = dict(zip(feature_names, importances))
        filtered_importance = {k: v for k, v in importance_dict.items() if
                               k not in ['lat', 'lon', 'extrap_lon', 'extrap_lat', 'lh']}

        # Sort the filtered importances for visualization
        sorted_features = sorted(filtered_importance, key=filtered_importance.get, reverse=True)
        sorted_importances = [filtered_importance[feature] for feature in sorted_features]

        # Example color coding: Top 3 features in green, rest in blue
        colors = ['green' if i < 3 else 'blue' for i in range(len(sorted_features))]

        if as_image == 0:
            data = {"title": "Filtered Feature Importances", "values_label": "Importance",
                    "data": [
                        {"label": f"{sorted_features[i]}", "value": f"{sorted_importances[i]}", "color": colors[i]}
                        for i in range(0, len(sorted_features))
                    ]}
            return data

        fig, ax = plt.subplots()
        y_pos = np.arange(len(sorted_features))
        ax.barh(y_pos, sorted_importances, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)
        ax.invert_yaxis()  # Highest importance at the top
        ax.set_xlabel('Importance')
        ax.set_title('Filtered Feature Importances')

        plt.tight_layout()

        return fig

    @app.route('/llm_providers', methods=['GET'])
    def get_llm_providers():
        """Get available LLM providers and their models"""
        api_keys = {
            "openai": OPENAI_API_KEY,
            "anthropic": ANTHROPIC_API_KEY,
            "google": GOOGLE_API_KEY,
            "cohere": COHERE_API_KEY
        }
        
        providers_info = LLMFactory.get_all_providers_info(api_keys)
        
        return {
            "providers": providers_info,
            "default_provider": DEFAULT_LLM_PROVIDER,
            "default_model": DEFAULT_LLM_MODEL,
            "default_temperature": DEFAULT_LLM_TEMPERATURE
        }

    @app.route('/chat_response', methods=['POST'])
    def chat_response():
        data = is_json(request.data)
        if data[0] is False: 
            return "Missing arguments.", 400
        
        # Get LLM parameters from request
        request_data = data[1]
        expl = request_data.get("explanation", "")
        provider_name = request_data.get("provider", DEFAULT_LLM_PROVIDER)
        model = request_data.get("model", DEFAULT_LLM_MODEL)
        temperature = request_data.get("temperature", DEFAULT_LLM_TEMPERATURE)
        max_tokens = request_data.get("max_tokens", 1000)
        
        return chat_response_func(expl, provider_name, model, temperature, max_tokens)

    def chat_response_func(expl, provider_name="openai", model="gpt-4-turbo-preview", temperature=0.0, max_tokens=1000):
        # Get API key for the provider
        api_keys = {
            "openai": OPENAI_API_KEY,
            "anthropic": ANTHROPIC_API_KEY,
            "google": GOOGLE_API_KEY,
            "cohere": COHERE_API_KEY
        }
        
        api_key = api_keys.get(provider_name)
        if not api_key:
            return f"Missing API key for provider: {provider_name}", 400
        
        try:
            # Create provider and generate response
            provider = LLMFactory.create_provider(provider_name, api_key)
            
            messages = [
                {"role": "system",
                 "content": "You are an expert data scientist with a talent for presenting to non-experts the results "
                            "of an AI model for trajectory forecasting."},
                {"role": "user",
                 "content": "Objective: Provide concise, user-friendly summaries of insights derived from XAI methods. "
                            "If multiple insights can be drawn from a single input try to combine them into a larger "
                            "context. Let's think step by step about how the final insight is reached. "
                            "Output Expectations: "
                            "Clarity: Deliver straightforward and easily understandable summaries. "
                            "Relevance: Ensure insights are directly applicable to the user's context or domain. "
                            "Actionability: Focus on providing practical suggestions or conclusions. "
                            "Responsiveness: Tailor summaries to answer user-specific inquiries based on the XAI "
                            "analysis. DO NOT include the input's obvious elements (numbers, text). Provide information"
                            " about how he should understand the input and the insight(s). Tailor responses to a user's"
                            " technical and domain expertise. Provide examples that match the expertise of the user in "
                            "an analogous manner to explain the insights. & Fully integrates all elements including "
                            "clarity, relevance, actionability, responsiveness, and user-specific customization, "
                            "creating a comprehensive and detailed approach to summarizing XAI insights. "
                            "The input = " + str(expl)
                 }
            ]
            
            response = provider.generate_response(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response, 200
            
        except Exception as e:
            return f"Error generating response: {str(e)}", 500

    @app.route('/index_instances', methods=['GET'])
    def index_instances():
        model_no = "a"; selected_tid = 0
        if request.args:
            arguments = request.args.to_dict()
            if "model" in arguments:
                model_no = arguments["model"]
                if model_no not in ["a", "b"]: model_no = "a"
            if "sid" not in arguments: return "Missing arguments.", 400
            selected_tid = int(arguments["sid"])

        df = pd.read_pickle(f'mobi_data/model_{model_no}_data.pkl')
        # # connector
        # data_url = f"{CONNECTOR_URL}/api/connector/csv/{model_no}"
        # response = requests.get(data_url)
        # response.raise_for_status()
        # csv_buffer = io.StringIO(response.content.decode("utf-8"))
        # df = pd.read_csv(csv_buffer)

        filtered_df = df[df['_tid'] == selected_tid].reset_index(drop=True)
        return {"min_value": int(filtered_df.index.min()), "max_value": int(filtered_df.index.max())}

    return app


if __name__ == "__main__":
    run_app()
