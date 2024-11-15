from flask import Blueprint, jsonify
from app.forms import MeasurementForm
from werkzeug.utils import secure_filename
import os

from app.services.calculation_service import CalculateMeasurement, Helper

# Blueprint initialization
measurement_bp = Blueprint("measurement", __name__)
UPLOAD_FOLDER = "uploads"

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Endpoint for uploading an image
@measurement_bp.route("/calculate", methods=["POST"])
def calculate_measurement():
    form = MeasurementForm()
    if form.validate_on_submit():
        # Get the validated data
        ori_width = float(form.ori_width.data)
        ori_height = float(form.ori_height.data)
        pxl_width = float(form.pxl_width.data)
        pxl_height = float(form.pxl_height.data)

        # Validate inputs
        if any(x <= 0 for x in [ori_width, ori_height, pxl_width, pxl_height]):
            return jsonify({"message": "Measurements must be positive numbers"}), 400

        image = form.image.data
        filename = secure_filename(image.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image.save(filepath)

        # Calculate the ratio (mm/pixel or units/pixel)
        ori_hyp = ori_height + ori_width
        pxl_hyp = pxl_height + pxl_width
        if pxl_hyp == 0:
            return jsonify({"message": "Pixel measurements cannot result in zero"}), 400
        ratio = ori_hyp / pxl_hyp

        print(f"Conversion ratio: {ratio:.4f} units/pixel")
        m=CalculateMeasurement()
        # Use the measurement service to calculate the measurements
        is_success,results = m.calculate(ratio, filepath)
        response_body = {
            "message": "Calculations were successful" if is_success else "Calculation failed",
            "result": results
        }
        return jsonify(response_body), 200 if is_success else 400
    else:
        return jsonify({"message": "Invalid form data", "errors": form.errors}), 400