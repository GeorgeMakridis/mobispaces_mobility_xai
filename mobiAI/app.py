import numpy as np
from flask import Flask, request, Response
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import pandas as pd
import geojson
import requests
import os

HOST_CONNECTOR = os.getenv("HOST_CONNECTOR")
PORT_CONNECTOR = os.getenv("PORT_CONNECTOR")
CONNECTOR_URL = f"{HOST_CONNECTOR}:{PORT_CONNECTOR}"


def run_app():
    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False
    app.config['JSON_SORT_KEYS'] = False

    @app.route('/spider_chart', methods=['POST'])
    def spider_chart():
        metrics = ['Clean (r/s x1000)', 'Augment (r/s x1000)', 'Compress (r/s x1000)']
        # values = bench()
        model2_values = [2.540, 0.771, 1.709]
        model3_values = [2.949, 0.765, 2.249]
        model4_values = [2.312, 0.586, 1.335]
        model5_values = [0.596, 0.162, 0.404]
        max_values = [3.1, 3.1, 3.1]  # Maximum possible values for normalization
        model_names = ['Current (Intel i9)', 'Apple M1', 'Intel i7', 'Raspberry Pi']
        title = "Platform Performance"
        data = {"metrics": metrics, "all_values": [model2_values, model3_values, model4_values, model5_values],
                "max_values": max_values, "title": title, "model_names": model_names}
        try:
            fig = plot_ai_performance_spider_chart(data["metrics"], data["all_values"], data["max_values"],
                                                   data["title"], data["model_names"])
            output = io.BytesIO()
            FigureCanvas(fig).print_png(output)
            return Response(output.getvalue(), mimetype='image/png')
        except Exception: return "There was a server error.", 500

    def plot_ai_performance_spider_chart(metrics, all_values, max_values, title="AI Model Performance", model_names=None):
        """
        Plot a spider chart for comparing multiple AI models across various metrics.
        """
        # Number of variables
        if model_names is None: model_names = []
        num_vars = len(metrics)

        # Calculate angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete loop

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
        fig.subplots_adjust(top=0.85, bottom=0.05)

        # Ensure model names are provided for each set of values
        if not model_names or len(model_names) != len(all_values):
            model_names = ['Model ' + str(i + 1) for i in range(len(all_values))]

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        # Draw one axe per variable and add labels
        plt.xticks(angles[:-1], metrics)

        for model_values, model_name in zip(all_values, model_names):
            # Normalize values
            normalized_values = [value / max_val for value, max_val in zip(model_values, max_values)]
            normalized_values += normalized_values[:1]  # Complete the loop to close the polygon
            # Plot
            ax.plot(angles, normalized_values, linewidth=2, linestyle='solid', label=model_name)
            ax.fill(angles, normalized_values, alpha=0.4)
        plt.title(title, size=20, y=1.1, x=0.2)
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.2))
        return fig

    # def bench():
    #     # Path to your Rust executable
    #     rust_executable_path = "pi_mt_marshal/target/x86_64-unknown-linux-gnu/release/marshal"
    #     # rust_executable_path = "/pi_mt_marshal/target/release/marshal"
    #
    #     # Run the Rust executable from Python
    #     result = subprocess.run([rust_executable_path], capture_output=True, text=True)
    #     # print(result)
    #     if result.returncode == 0:  # Check the result
    #         # print("Rust executable ran successfully.\n"Output:")
    #         print(result.stdout)
    #     else:
    #         print("Error running Rust executable.\nError message:")
    #         print(result.stderr)
    #     return [int(1e+9/float(val)) for val in result.stdout.split(',')]

    @app.route('/trips', methods=['GET'])
    def trips():
        model_no = "a"
        if request.args:
            arguments = request.args.to_dict()
            if "model" in arguments:
                model_no = arguments["model"]
                if model_no not in ["a", "b"]: model_no = "a"

        df = pd.read_pickle(f'/mobi_data/model_{model_no}_data.pkl')
        # # connector
        # data_url = f"{CONNECTOR_URL}/api/connector/csv/{model_no}"
        # response = requests.get(data_url)
        # response.raise_for_status()
        # csv_buffer = io.StringIO(response.content.decode("utf-8"))
        # df = pd.read_csv(csv_buffer)

        # Dropdown for selecting Trip ID
        tid_lst = df.loc[df.pred == 1].groupby('_tid').apply(lambda a: a.grpres.mean()).sort_values().index.tolist()[:10]
        if model_no == "a":
            tid_lst.remove(283); tid_lst.remove(93)
            tid_lst.insert(0, 283); tid_lst.insert(0, 93)
        elif model_no == "b":
            tid_lst.remove(313); tid_lst.insert(0, 313)
        max_value = 30 if model_no == "b" else 20
        return {"trip_ids": tid_lst, "max_value": max_value}, 200

    @app.route('/map_data', methods=['GET'])
    def map_data():
        model_no = "a"; selected_tid = 0; sel_lh = 0; as_geojson = 0
        if request.args:
            arguments = request.args.to_dict()
            if "model" in arguments:
                model_no = arguments["model"]
                if model_no not in ["a", "b"]: model_no = "a"
            if "sid" not in arguments: return "Missing arguments.", 400
            selected_tid = int(arguments["sid"])
            if "slh" not in arguments: return "Missing arguments.", 400
            sel_lh = int(arguments["slh"])
            if "geojson" in arguments: as_geojson = 1

        df = pd.read_pickle(f'mobi_data/model_{model_no}_data.pkl')
        # # connector
        # data_url = f"{CONNECTOR_URL}/api/connector/csv/{model_no}"
        # response = requests.get(data_url)
        # response.raise_for_status()
        # csv_buffer = io.StringIO(response.content.decode("utf-8"))
        # df = pd.read_csv(csv_buffer)

        # Filter the DataFrame for the selected trip ID to get unique 'lh' values for the second dropdown
        unique_lh_values = df[df['_tid'] == selected_tid]['lh'].unique()
        unique_lh_values.sort()  # Sort the values for better usability
        # Preparing lh values for the selected Trip ID
        unique_lh_values = sorted(df[df['_tid'] == selected_tid]['lh'].unique())
        sel_lh = unique_lh_values[int(sel_lh / 5)]
        # Filter the DataFrame based on '_tid' and 'lh'
        sdf = df[(df['_tid'] == selected_tid)].reset_index(drop=True)
        # Add a column to differentiate actual points from predictions
        sdf['pred_text'] = sdf['pred'].apply(lambda x: 'Prediction' if x == 1 else 'Actual')
        sdf = sdf[["index", "lon", "lat", "lh", "pred", "pred_text"]]
        # Fix: Replace original index column with reset index to match explanation system
        sdf['index'] = sdf.index
        if as_geojson == 0:
            return {"lat": sdf['lat'].mean(), "lon": sdf['lon'].mean(),
                    "actual_points": sdf[sdf['pred'] == 0].to_json(orient='split', index=False),
                    "predicted_points": sdf[(sdf['pred'] == 1) & (sdf['lh'] == sel_lh)].to_json(orient='split', index=False)}, 200
        features = []
        type_to_color = {"Actual": "yellow", "Prediction": "blue"}
        for _, row in sdf.iterrows():
            feature = geojson.Feature(
                geometry=geojson.Point((row["lon"], row["lat"])),
                properties={"index": row.name, "type": row["pred_text"],  # Fix: Use reset index instead of original index
                            "color": type_to_color[row["pred_text"]],
                            "marker-color": type_to_color[row["pred_text"]],
                            "longitude": row["lon"], "latitude": row["lat"]
                            }
            )
            features.append(feature)
        # Create a GeoJSON FeatureCollection
        return geojson.FeatureCollection(features), 200

    return app


if __name__ == "__main__":
    run_app()
