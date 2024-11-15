from flask_wtf import FlaskForm
from wtforms import FloatField, FileField
from wtforms.validators import DataRequired, NumberRange
from flask import request
import os

class MeasurementForm(FlaskForm):
    ori_width = FloatField('Original Width', validators=[DataRequired(), NumberRange(min=0)])
    ori_height = FloatField('Original Height', validators=[DataRequired(), NumberRange(min=0)])
    pxl_width = FloatField('Pixel Width', validators=[DataRequired(), NumberRange(min=0)])
    pxl_height = FloatField('Pixel Height', validators=[DataRequired(), NumberRange(min=0)])
    image = FileField('Image', validators=[DataRequired()])

    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

    def validate_image(self, field):
        """Custom validator to ensure the image has a valid extension."""
        if field.data:
            filename = field.data.filename
            if not filename.lower().endswith(tuple(self.ALLOWED_EXTENSIONS)):
                raise ValueError(f"Invalid image file type. Allowed types: {self.ALLOWED_EXTENSIONS} ")
