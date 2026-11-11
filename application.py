import os
from threading import Thread

from flask import Flask, request, render_template,jsonify
from flask_cors import CORS,cross_origin
from src.pipeline.model_comparison import get_model_comparison_metrics
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.retrain_pipeline import retrain_from_db
from src.database.db_config import init_db
from src.database.services import DashboardService, PredictionService

application = Flask(__name__)

app = application

def _init_db_async():
    try:
        init_db()
    except Exception as init_error:
        app.logger.exception("Database init failed: %s", init_error)


Thread(target=_init_db_async, daemon=True).start()

allowed_origins_env = os.getenv('ALLOWED_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173')
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(',') if origin.strip()]
CORS(app, resources={r'/*': {'origins': allowed_origins}})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/')
@cross_origin()
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
@cross_origin()
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            age = int(request.form.get('age')),
            gender = request.form.get('gender'),
            subscription_type = request.form.get('subscription_type'),
            watch_hours = float(request.form.get('watch_hours')),
            last_login_days = int(request.form.get('last_login_days')),
            region = request.form.get('region'),
            device = request.form.get('device'),
            monthly_fee = float(request.form.get('monthly_fee')),
            payment_method = request.form.get('payment_method'),
            number_of_profiles = int(request.form.get('number_of_profiles')),
            avg_watch_time_per_day = float(request.form.get('avg_watch_time_per_day')),
            favorite_genre = request.form.get('favorite_genre')
        )

        pred_df = data.get_data_as_dataframe()

        print(pred_df)

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)
        results = 'Churned' if int(pred['prediction']) == 1 else 'Not Churned'

        try:
            PredictionService.save_prediction(
                age=data.age,
                gender=data.gender,
                subscription_type=data.subscription_type,
                watch_hours=data.watch_hours,
                last_login_days=data.last_login_days,
                region=data.region,
                device=data.device,
                monthly_fee=data.monthly_fee,
                payment_method=data.payment_method,
                number_of_profiles=data.number_of_profiles,
                avg_watch_time_per_day=data.avg_watch_time_per_day,
                favorite_genre=data.favorite_genre,
                prediction=int(pred['prediction']),
                prediction_status=results,
                prediction_probability=pred['probability'],
                source='form',
                ip_address=request.remote_addr,
            )
        except Exception as save_error:
            app.logger.exception("Failed to persist prediction: %s", save_error)

        return render_template('index.html',results=results,pred_df = pred_df)
    
@app.route('/predictAPI',methods=['POST'])
@cross_origin()
def predict_api():
    if request.method=='POST':
        data = CustomData(
            age = int(request.json['age']),
            gender = request.json['gender'],
            subscription_type = request.json['subscription_type'],
            watch_hours = float(request.json['watch_hours']),
            last_login_days = int(request.json['last_login_days']),
            region = request.json['region'],
            device = request.json['device'],
            monthly_fee = float(request.json['monthly_fee']),
            payment_method = request.json['payment_method'],
            number_of_profiles = int(request.json['number_of_profiles']),
            avg_watch_time_per_day = float(request.json['avg_watch_time_per_day']),
            favorite_genre = request.json['favorite_genre']
        )

        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)

        results = 'Churned' if int(pred['prediction']) == 1 else 'Not Churned'

        try:
            PredictionService.save_prediction(
                age=data.age,
                gender=data.gender,
                subscription_type=data.subscription_type,
                watch_hours=data.watch_hours,
                last_login_days=data.last_login_days,
                region=data.region,
                device=data.device,
                monthly_fee=data.monthly_fee,
                payment_method=data.payment_method,
                number_of_profiles=data.number_of_profiles,
                avg_watch_time_per_day=data.avg_watch_time_per_day,
                favorite_genre=data.favorite_genre,
                prediction=int(pred['prediction']),
                prediction_status=results,
                prediction_probability=pred['probability'],
                source='api',
                ip_address=request.remote_addr,
            )
        except Exception as save_error:
            app.logger.exception("Failed to persist prediction: %s", save_error)

        dct = {
            'churned': int(pred['prediction']),
            'status': results,
            'probability': pred['probability'],
        }
        return jsonify(dct)


@app.route('/modelComparisonAPI', methods=['GET'])
@cross_origin()
def model_comparison_api():
    comparison = get_model_comparison_metrics()
    return jsonify(comparison)


@app.route('/dashboardStatsAPI', methods=['GET'])
@cross_origin()
def dashboard_stats_api():
    stats = DashboardService.get_dashboard_stats()
    return jsonify(stats)


@app.route('/retrainAPI', methods=['POST'])
@cross_origin()
def retrain_api():
    try:
        result = retrain_from_db()
        return jsonify(result), 200
    except Exception as exc:
        app.logger.exception("Retrain failed: %s", exc)
        return jsonify({
            'status': 'error',
            'reason': 'Retrain failed on the server. Check logs for details.'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))