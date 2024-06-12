from flask import Flask
import logging
from config.logging_config import setup_logging

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)

    from app.routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
