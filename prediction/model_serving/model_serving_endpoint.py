from os import getenv
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement, dict_factory
from cassandra.protocol import ProtocolException

from operator import itemgetter

from flask import (render_template as rt,
                   Flask, request, redirect, url_for, session, jsonify)
from flask_cors import CORS

from gevent.pywsgi import WSGIServer

import jwt
import re
from datetime import datetime

"""
    Database connector
"""


class Cassandra:
    def __init__(self):
        self.MORPHL_SERVER_IP_ADDRESS = getenv('MORPHL_SERVER_IP_ADDRESS')
        self.MORPHL_CASSANDRA_USERNAME = getenv('MORPHL_CASSANDRA_USERNAME')
        self.MORPHL_CASSANDRA_PASSWORD = getenv('MORPHL_CASSANDRA_PASSWORD')
        self.MORPHL_CASSANDRA_KEYSPACE = getenv('MORPHL_CASSANDRA_KEYSPACE')

        self.CASS_REQ_TIMEOUT = 3600.0

        self.auth_provider = PlainTextAuthProvider(
            username=self.MORPHL_CASSANDRA_USERNAME,
            password=self.MORPHL_CASSANDRA_PASSWORD)
        self.cluster = Cluster(
            [self.MORPHL_SERVER_IP_ADDRESS], auth_provider=self.auth_provider)

        self.session = self.cluster.connect(self.MORPHL_CASSANDRA_KEYSPACE)
        self.session.row_factory = dict_factory
        self.session.default_fetch_size = 100

        self.prepare_statements()

    def prepare_statements(self):
        """
            Prepare statements for database select queries
        """
        self.prep_stmts = {
            'predictions': {},
            'access_logs': {}
        }

        template_for_single_row = 'SELECT * FROM ga_epna_predictions WHERE client_id = ? LIMIT 1'
        template_for_access_log_insert = 'INSERT INTO ga_epna_predictions_access_logs (client_id, tstamp, all_visits, product_view, add_to_cart, checkout_with_add_to_cart, checkout_without_add_to_cart, transaction) VALUES (?,?,?,?,?,?,?,?)'

        self.prep_stmts['predictions']['single'] = self.session.prepare(
            template_for_single_row)
        self.prep_stmts['access_logs']['insert'] = self.session.prepare(
            template_for_access_log_insert)

    def retrieve_prediction(self, client_id):
        bind_list = [client_id]
        return self.session.execute(self.prep_stmts['predictions']['single'], bind_list, timeout=self.CASS_REQ_TIMEOUT)._current_rows

    def insert_access_log(self, client_id, p):
        if len(p) == 0:
            bind_list = [client_id, datetime.now(), -1, -1, -1, -1, -1, -1]
        else:
            bind_list = [client_id, datetime.now(), p[0]['all_visits'], p[0]['product_view'], p[0]['add_to_cart'],
                         p[0]['checkout_with_add_to_cart'], p[0]['checkout_without_add_to_cart'], p[0]['transaction']]

        return self.session.execute(self.prep_stmts['access_logs']['insert'],
                                    bind_list, timeout=self.CASS_REQ_TIMEOUT)


"""
    API class for verifying credentials and handling JWTs.
"""


class API:
    def __init__(self):
        self.API_DOMAIN = getenv('API_DOMAIN')
        self.MORPHL_API_KEY = getenv('MORPHL_API_KEY')
        self.MORPHL_API_JWT_SECRET = getenv('MORPHL_API_JWT_SECRET')

    def verify_jwt(self, token):
        try:
            decoded = jwt.decode(token, self.MORPHL_API_JWT_SECRET)
        except Exception:
            return False

        return (decoded['iss'] == self.API_DOMAIN and
                decoded['sub'] == self.MORPHL_API_KEY)


app = Flask(__name__)
CORS(app)

# @todo Check request origin for all API requests


@app.route("/shopping-stage")
def main():
    return "MorphL Predictions API - Shopping Stage"


@app.route('/shopping-stage/getprediction/<client_id>')
def get_prediction(client_id):
    # Validate authorization header with JWT
    if request.headers.get('Authorization') is None or not app.config['API'].verify_jwt(request.headers['Authorization']):
        return jsonify(status=0, error='Unauthorized request.'), 401

    # Validate client id (alphanumeric with dots)
    if not re.match('^[a-zA-Z0-9.]+$', client_id):
        return jsonify(status=0, error='Invalid client id.')

    p = app.config['CASSANDRA'].retrieve_prediction(client_id)

    # Log prediction request
    app.config['CASSANDRA'].insert_access_log(client_id, p)

    if len(p) == 0:
        return jsonify(status=0, error='No associated predictions found for that ID.')

    del p[0]['client_id']
    return jsonify(status=1, prediction={'client_id': client_id, 'shopping_stages': {
        'all_visits': round(p[0]['all_visits'], 4),
        'product_view': round(p[0]['product_view'], 4),
        'add_to_cart': round(p[0]['add_to_cart'], 4),
        'checkout_with_add_to_cart': round(p[0]['checkout_with_add_to_cart'], 4),
        'checkout_without_add_to_cart': round(p[0]['checkout_without_add_to_cart'], 4),
        'transaction': round(p[0]['transaction'], 4),
        'prediction_date': str(p[0]['prediction_date'])
    }})


if __name__ == '__main__':
    app.config['CASSANDRA'] = Cassandra()
    app.config['API'] = API()
    if getenv('DEBUG'):
        app.config['DEBUG'] = True
        flask_port = 5858
        app.run(host='0.0.0.0', port=flask_port)
    else:
        app.config['DEBUG'] = False
        flask_port = 6868
        WSGIServer(('', flask_port), app).serve_forever()
