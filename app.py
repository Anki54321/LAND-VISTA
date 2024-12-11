from flask import Flask, request, jsonify, render_template
import torch
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Initialize Flask app
app = Flask(__name__)

# Define the PricePredictor class
class PricePredictor(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PricePredictor, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # Direct output without ReLU for more control

# Load the trained PyTorch model (dummy weights for this example)
model = PricePredictor(input_dim=3, output_dim=3)
model.load_state_dict(torch.load(r"C:\Users\Anki\Desktop\lv\Dev Process\model\price_predictor_model.pth",weights_only=True))
 # Update with correct path
model.eval()
# Load and preprocess the dataset
df = pd.read_csv(r"C:\Users\Anki\Desktop\lv\Dev Process\converted_data.csv")  # Replace with the correct path to your dataset

# Initialize LabelEncoders and Scaler
label_encoder_district = LabelEncoder()
label_encoder_region = LabelEncoder()
scaler = MinMaxScaler()

# Fit LabelEncoders and Scaler
df['District_Encoded'] = label_encoder_district.fit_transform(df['District'])
df['Region_Encoded'] = label_encoder_region.fit_transform(df['Region'])
scaler.fit(df[['Average_Price', 'Min_Price', 'Max_Price']])

# Helper function to encode inputs
def encode_inputs(year, district, region):
    district_encoded = label_encoder_district.transform([district])[0]
    region_encoded = label_encoder_region.transform([region])[0]
    input_array = np.array([[year, district_encoded, region_encoded]])
    return torch.tensor(input_array, dtype=torch.float32)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/agent")
def agent():
    return render_template("agent.html")

@app.route("/properties_single")
def properties_single():
    return render_template("properties-single.html")

@app.route("/blog_single")
def blog_single():
    return render_template("blog-single.html")

@app.route("/services")
def services():
    return render_template("services.html")

@app.route("/blog")
def blog():
    return render_template("blog.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/properties")
def properties():
    return render_template("properties.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input parameters
        year = int(request.form["year"])
        district = request.form["district"]
        region = request.form["region"]

        # Validate district and region inputs
        if district not in label_encoder_district.classes_ or region not in label_encoder_region.classes_:
            return jsonify({"error": "Invalid district or region"}), 400

        # Filter historical data
        filtered_data = df[(df["District"] == district) & (df["Region"] == region)]
        historical_data = filtered_data[filtered_data["Year"] < year]
        historical_data = historical_data.sort_values(by="Year")

        # Generate increasing predictions
        if not historical_data.empty:
            last_known_price = historical_data["Average_Price"].iloc[-1]
        else:
            last_known_price = df["Average_Price"].mean()  # Default to mean if no history

        predicted_avg = last_known_price * 1.05  # Increase by 5%
        predicted_min = predicted_avg * 0.8  # Min price is 80% of avg
        predicted_max = predicted_avg * 1.2  # Max price is 120% of avg

        # Append prediction to historical data
        prediction_data = pd.DataFrame({"Year": [year], "Average_Price": [predicted_avg]})
        combined_data = pd.concat([historical_data[["Year", "Average_Price"]], prediction_data], ignore_index=True)

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(combined_data["Year"], combined_data["Average_Price"], marker="o", label="Historical Prices")
        plt.scatter(year, predicted_avg, color="red", label="Predicted Average Price", zorder=5)
        plt.axvline(year, color="gray", linestyle="--", label="Prediction Year")
        plt.title(f"Price Trends in {district}, {region}", fontsize=16)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Average Price", fontsize=12)
        plt.legend()
        plt.grid()
        plt.ylim(0, 1.2 * max(combined_data["Average_Price"].max(), predicted_avg))

        # Convert plot to a string
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode("utf8")
        buf.close()

        # Return prediction and plot
        return render_template(
            "prediction.html",
            predicted_avg=f"{predicted_avg:.2f}",
            predicted_min=f"{predicted_min:.2f}",
            predicted_max=f"{predicted_max:.2f}",
            plot_url=f"data:image/png;base64,{plot_url}"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
