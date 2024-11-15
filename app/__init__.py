#app/__init__.py
from flask import Flask
from app.routes import measurement_bp
from app.utils.loader import Loader

def create_app():
    app = Flask(__name__)
    app.config['WTF_CSRF_ENABLED'] = False
    app.register_blueprint(measurement_bp)
    with app.app_context():
        app.config['MODEL'] = Loader.load_model()
        app.config['CONFIG_PROPERTIES'] = Loader.load_garment_config()
    return app
