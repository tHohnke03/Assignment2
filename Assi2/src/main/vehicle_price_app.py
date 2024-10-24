import tkinter as tk
from tkinter import messagebox
import joblib


class VehiclePriceApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title("Vehicle Price Prediction App")

        self.model = model

        # Define input fields for the features used in the model
        self.feature_labels = []
        self.feature_entries = []
        self.create_input_fields()

        # Add a predict button
        self.predict_button = tk.Button(root, text="Predict", command=self.get_prediction)
        self.predict_button.grid(row=len(self.feature_entries) + 1, column=1, padx=10, pady=20)

    def create_input_fields(self):
        """Dynamically create input fields for each feature."""
        feature_names = ['Year', 'Brand', 'Model', 'Car/Suv', 'UsedOrNew', 'Transmission', 'Engine',
                         'DriveType', 'FuelType', 'FuelConsumption', 'Kilometres', 'ColourExtInt',
                         'Location', 'CylindersinEngine', 'BodyType', 'Doors', 'Seats']

        for i, feature in enumerate(feature_names):
            label = tk.Label(self.root, text=feature)
            label.grid(row=i, column=0, padx=10, pady=5)
            entry = tk.Entry(self.root)
            entry.grid(row=i, column=1, padx=10, pady=5)

            self.feature_labels.append(label)
            self.feature_entries.append(entry)

    def get_prediction(self):
        """Get input values from the entries, make a prediction, and display the result."""
        try:
            # Define which features should be numeric
            numeric_features = ['Year', 'FuelConsumption', 'Kilometres', 'CylindersinEngine', 'Doors', 'Seats']

            features = []
            feature_names = ['Year', 'Brand', 'Model', 'Car/Suv', 'UsedOrNew', 'Transmission', 'Engine',
                             'DriveType', 'FuelType', 'FuelConsumption', 'Kilometres', 'ColourExtInt',
                             'Location', 'CylindersinEngine', 'BodyType', 'Doors', 'Seats']

            for i, entry in enumerate(self.feature_entries):
                input_value = entry.get().strip()
                if input_value == "":
                    raise ValueError(f"Please fill out the {feature_names[i]} field.")

                # Convert numeric fields to integers or floats
                if feature_names[i] in numeric_features:
                    try:
                        features.append(float(input_value))  # Handle numeric conversion
                    except ValueError:
                        raise ValueError(f"The {feature_names[i]} field should be a number.")
                else:
                    features.append(input_value)  # Keep other fields as strings

            # Make a prediction using the model (assuming the model accepts a list of mixed data types)
            prediction_result = self.model.predict([features])

            # Display the result
            messagebox.showinfo("Prediction", f"The predicted vehicle price is: ${prediction_result[0]:.2f}")

        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Something went wrong: {str(e)}")


if __name__ == "__main__":

    try:
        # Load the best model
        model = joblib.load("/Users/trenthohnke/Downloads/Assi2/best_model.pkl")
    except FileNotFoundError:
        messagebox.showerror("Error", "Model file not found. Please check the path.")
        model = None  # Initialize model as None to prevent passing an uninitialized model

    if model:  # Only run the GUI if the model is loaded successfully
        root = tk.Tk()
        app = VehiclePriceApp(root, model)
        root.mainloop()
