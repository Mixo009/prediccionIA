    import os
import json
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

# Etiquetas de diagnóstico
DIAGNOSIS_LABELS = {1: 'Dengue', 2: 'Malaria', 3: 'Leptospirosis'}

# Definición del esquema (55 variables predictoras) según docs
FEATURE_SECTIONS = [
    {
        'name': 'Demografía',
        'fields': [
            {'name': 'age', 'type': 'number'},
            {'name': 'male', 'type': 'binary'},
            {'name': 'female', 'type': 'binary'},
            {'name': 'urban_origin', 'type': 'binary'},
            {'name': 'rural_origin', 'type': 'binary'},
        ],
    },
    {
        'name': 'Ocupación',
        'fields': [
            {'name': 'homemaker', 'type': 'binary'},
            {'name': 'student', 'type': 'binary'},
            {'name': 'professional', 'type': 'binary'},
            {'name': 'merchant', 'type': 'binary'},
            {'name': 'agriculture_livestock', 'type': 'binary'},
            {'name': 'various_jobs', 'type': 'binary'},
            {'name': 'unemployed', 'type': 'binary'},
        ],
    },
    {
        'name': 'Clínicos',
        'fields': [
            {'name': 'hospitalization_days', 'type': 'number'},
            {'name': 'body_temperature', 'type': 'number'},
        ],
    },
    {
        'name': 'Síntomas',
        'fields': [
            {'name': 'fever', 'type': 'binary'},
            {'name': 'headache', 'type': 'binary'},
            {'name': 'dizziness', 'type': 'binary'},
            {'name': 'loss_of_appetite', 'type': 'binary'},
            {'name': 'weakness', 'type': 'binary'},
            {'name': 'myalgias', 'type': 'binary'},
            {'name': 'arthralgias', 'type': 'binary'},
            {'name': 'eye_pain', 'type': 'binary'},
            {'name': 'hemorrhages', 'type': 'binary'},
            {'name': 'vomiting', 'type': 'binary'},
            {'name': 'abdominal_pain', 'type': 'binary'},
            {'name': 'chills', 'type': 'binary'},
            {'name': 'edema', 'type': 'binary'},
            {'name': 'jaundice', 'type': 'binary'},
            {'name': 'bruises', 'type': 'binary'},
            {'name': 'petechiae', 'type': 'binary'},
            {'name': 'rash', 'type': 'binary'},
            {'name': 'diarrhea', 'type': 'binary'},
            {'name': 'respiratory_difficulty', 'type': 'binary'},
            {'name': 'itching', 'type': 'binary'},
        ],
    },
    {
        'name': 'Laboratorio',
        'fields': [
            {'name': 'hematocrit', 'type': 'number'},
            {'name': 'hemoglobin', 'type': 'number'},
            {'name': 'red_blood_cells', 'type': 'number'},
            {'name': 'white_blood_cells', 'type': 'number'},
            {'name': 'neutrophils', 'type': 'number'},
            {'name': 'eosinophils', 'type': 'number'},
            {'name': 'basophils', 'type': 'number'},
            {'name': 'monocytes', 'type': 'number'},
            {'name': 'lymphocytes', 'type': 'number'},
            {'name': 'platelets', 'type': 'number'},
            {'name': 'AST', 'type': 'number'},
            {'name': 'ALT', 'type': 'number'},
            {'name': 'ALP', 'type': 'number'},
            {'name': 'total_bilirubin', 'type': 'number'},
            {'name': 'direct_bilirubin', 'type': 'number'},
            {'name': 'indirect_bilirubin', 'type': 'number'},
            {'name': 'total_proteins', 'type': 'number'},
            {'name': 'albumin', 'type': 'number'},
            {'name': 'creatinine', 'type': 'number'},
            {'name': 'urea', 'type': 'number'},
        ],
    },
]

def feature_columns():
    return [f['name'] for sec in FEATURE_SECTIONS for f in sec['fields']]

# Artefactos en memoria
MODELS = {
    'logistic': None,
    'nn': None,
}
TRAINING_INFO = {
    'logistic': None,
    'nn': None,
}

def normalize_binary(v):
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ('1','true','si','sí','on','y','yes'): return 1
        if s in ('0','false','no','off','n'): return 0
        try:
            return float(s)
        except Exception:
            return 0
    return int(v) if isinstance(v, (bool, int)) else (float(v) if isinstance(v, float) else 0)

def prepare_df(df):
    cols = feature_columns()
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    # binarios
    binary_cols = [f['name'] for sec in FEATURE_SECTIONS for f in sec['fields'] if f['type']=='binary']
    for c in binary_cols:
        df[c] = df[c].apply(normalize_binary)
    # numéricos: convertir
    numeric_cols = [f['name'] for sec in FEATURE_SECTIONS for f in sec['fields'] if f['type']=='number']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    return df[cols]

def init_models():
    data_path = os.path.join('data', 'DEMALE-HSJM_2025_data.xlsx')
    if not os.path.exists(data_path):
        print(f'[ERROR] No se encontró el archivo de datos en: {os.path.abspath(data_path)}')
        print(f'[ERROR] Directorio actual: {os.getcwd()}')
        print(f'[ERROR] Archivos en data/: {os.listdir("data") if os.path.exists("data") else "Directorio data/ no existe"}')
        return
    try:
        df = pd.read_excel(data_path)
    except Exception as e:
        print(f'[ERROR] No se pudo leer el archivo Excel: {str(e)}')
        return
    if 'diagnosis' not in df.columns:
        print(f'[ERROR] El archivo no contiene la columna "diagnosis". Columnas encontradas: {list(df.columns)}')
        return
    y = df['diagnosis'].astype(int)
    X = prepare_df(df.drop(columns=['diagnosis']))

    # Separar en train y test para evitar sobreajuste
    # SMOTE solo se aplica en el conjunto de entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Aplicar SMOTE solo en el conjunto de entrenamiento
    print('[ML] Aplicando SMOTE al conjunto de entrenamiento...')
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f'[ML] Dataset original: {len(X_train)} muestras')
    print(f'[ML] Dataset balanceado con SMOTE: {len(X_train_balanced)} muestras')
    print(f'[ML] Distribución balanceada: {pd.Series(y_train_balanced).value_counts().to_dict()}')
    
    # Guardar información de distribución para mostrar en frontend
    original_dist = pd.Series(y).value_counts().to_dict()
    balanced_dist_train = pd.Series(y_train_balanced).value_counts().to_dict()
    X_all_balanced_temp, y_all_balanced_temp = smote.fit_resample(X, y)
    balanced_dist_all = pd.Series(y_all_balanced_temp).value_counts().to_dict()

    # Validación cruzada estratificada para respetar el balance por clase
    # Usar más folds para mejor evaluación en dataset pequeño
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Pipeline + GridSearchCV para Regresión Logística
    pipe_log = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(max_iter=1000, multi_class='ovr')),
    ])
    grid_log = {
        'logistic__C': [0.01, 0.1, 1, 10, 100],
        'logistic__penalty': ['l2'],
        'logistic__solver': ['liblinear'],
        'logistic__class_weight': ['balanced'],
    }
    gs_log = GridSearchCV(pipe_log, grid_log, cv=cv, scoring='balanced_accuracy', n_jobs=1)
    gs_log.fit(X_train_balanced, y_train_balanced)
    
    # Evaluar en el conjunto de test original (sin SMOTE) para evitar sobreajuste
    test_score = gs_log.score(X_test, y_test)
    print('[ML] Logistic best params:', gs_log.best_params_)
    print('[ML] Logistic CV balanced_accuracy (train):', round(gs_log.best_score_, 4))
    print('[ML] Logistic balanced_accuracy (test):', round(test_score, 4))
    
    # Entrenar modelo final con todos los datos balanceados
    final_model_log = gs_log.best_estimator_
    # Re-entrenar con todos los datos balanceados para producción
    X_all_balanced, y_all_balanced = smote.fit_resample(X, y)
    final_model_log.fit(X_all_balanced, y_all_balanced)
    MODELS['logistic'] = final_model_log
    
    TRAINING_INFO['logistic'] = {
        'cv_n_splits': cv.get_n_splits(),
        'cv_metric': 'balanced_accuracy',
        'cv_best_score': float(gs_log.best_score_),
        'test_score': float(test_score),
        'best_params': gs_log.best_params_,
        'smote_applied': True,
        'original_distribution': {int(k): int(v) for k, v in original_dist.items()},
        'balanced_distribution': {int(k): int(v) for k, v in balanced_dist_all.items()},
        'original_total': len(X),
        'balanced_total': len(X_all_balanced_temp),
    }

    pipe_mlp = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            max_iter=1500,
            batch_size=16,
            learning_rate_init=0.001,
            solver='adam',  # Adam funciona mejor con early_stopping
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,  # Más datos para validación temprana
            n_iter_no_change=20,  # Más paciencia antes de parar
            warm_start=False  # Reiniciar pesos en cada fit
        )),
    ])
    grid_mlp = {
        # Arquitecturas optimizadas para balanced accuracy (priorizar clases minoritarias)
        # Capas medianas que capturan patrones sin sobreajuste
        'mlp__hidden_layer_sizes': [(32,), (32, 16), (24, 12), (40, 20)],
        # Regularización ajustada para mejor balance (no demasiado alta para no perder patrones)
        'mlp__alpha': [0.01, 0.05, 0.1, 0.2],
        # Tasa de aprendizaje adaptativa generalmente funciona mejor para balanced accuracy
        'mlp__learning_rate': ['adaptive'],  # Focus en adaptive para mejor convergencia
        # Funciones de activación que funcionan mejor con clases desbalanceadas
        'mlp__activation': ['tanh', 'logistic'],  # Suaves y continuas, mejor para datos pequeños
        # Tasas de aprendizaje más conservadoras para mejor convergencia
        'mlp__learning_rate_init': [0.0005, 0.001, 0.002],
        # Tolerancia más estricta para mejor convergencia y precisión
        'mlp__tol': [1e-5],
    }
    gs_mlp = GridSearchCV(pipe_mlp, grid_mlp, cv=cv, scoring='balanced_accuracy', n_jobs=1)
    gs_mlp.fit(X_train_balanced, y_train_balanced)
    
    # Evaluar en el conjunto de test original (sin SMOTE) para evitar sobreajuste
    test_score_mlp = gs_mlp.score(X_test, y_test)
    print('[ML] MLP best params:', gs_mlp.best_params_)
    print('[ML] MLP CV balanced_accuracy (train):', round(gs_mlp.best_score_, 4))
    print('[ML] MLP balanced_accuracy (test):', round(test_score_mlp, 4))
    
    # Entrenar modelo final con todos los datos balanceados
    final_model_mlp = gs_mlp.best_estimator_
    # Re-entrenar con todos los datos balanceados para producción
    X_all_balanced_mlp, y_all_balanced_mlp = smote.fit_resample(X, y)
    final_model_mlp.fit(X_all_balanced_mlp, y_all_balanced_mlp)
    MODELS['nn'] = final_model_mlp
    
    TRAINING_INFO['nn'] = {
        'cv_n_splits': cv.get_n_splits(),
        'cv_metric': 'balanced_accuracy',
        'cv_best_score': float(gs_mlp.best_score_),
        'test_score': float(test_score_mlp),
        'best_params': gs_mlp.best_params_,
        'smote_applied': True,
        'original_distribution': {int(k): int(v) for k, v in original_dist.items()},
        'balanced_distribution': {int(k): int(v) for k, v in balanced_dist_all.items()},
        'original_total': len(X),
        'balanced_total': len(X_all_balanced_temp),
    }

# Inicializar modelos al cargar la aplicación (se ejecuta siempre, incluso con gunicorn)
init_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/schema')
def schema():
    return jsonify(success=True, sections=FEATURE_SECTIONS, labels=DIAGNOSIS_LABELS)

@app.route('/individual')
def individual():
    return render_template('individual.html')

@app.route('/batch')
def batch():
    return render_template('batch.html')

@app.route('/predict_individual', methods=['POST'])
def predict_individual():
    # Extraer el tipo de modelo primero
    if request.is_json:
        model_type = request.json.get('model', 'logistic')
        payload = request.json.copy()
    else:
        model_type = request.form.get('model', 'logistic')
        payload = request.form.to_dict()
    
    # Validar y limpiar el payload (eliminar 'model' si existe)
    if 'model' in payload:
        del payload['model']
    
    model_type = model_type if model_type in ('logistic','nn') else 'logistic'
    model = MODELS.get(model_type)
    
    if model is None:
        return jsonify(success=False, error=f'Modelo {model_type} no inicializado'), 500
   
    print(f'[DEBUG] Usando modelo: {model_type}')
    
    # Datos
    df = pd.DataFrame([payload])
    X = prepare_df(df)
    probs = model.predict_proba(X)[0]
    pred = int(np.argmax(probs) + 1)
    
    return jsonify(
        success=True,
        diagnosis=pred,
        diagnosis_label=DIAGNOSIS_LABELS.get(pred, f'Clase {pred}'),
        probabilities={1: float(probs[0]), 2: float(probs[1]), 3: float(probs[2])},
        model_info=TRAINING_INFO.get(model_type),
        model_used=model_type  # Agregar para verificación en frontend
    )

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    model_type = request.form.get('model', 'logistic')
    model_type = model_type if model_type in ('logistic','nn') else 'logistic'
    model = MODELS.get(model_type)
    
    if model is None:
        return jsonify(success=False, error=f'Modelo {model_type} no inicializado'), 500
    
    # Debug: verificar que se está usando el modelo correcto
    print(f'[DEBUG] Usando modelo: {model_type}')
    
    # Verificar que se envió un archivo
    if 'file' not in request.files:
        return jsonify(success=False, error='No se adjuntó archivo'), 400
    
    file = request.files['file']
    
    # Verificar que el archivo tiene nombre (no está vacío)
    if file.filename == '' or not file.filename:
        return jsonify(success=False, error='No se seleccionó ningún archivo'), 400
    
    print(f'[DEBUG] Archivo recibido: {file.filename}')
    
    try:
        # Leer el archivo según su extensión
        filename_lower = file.filename.lower()
        if filename_lower.endswith('.xlsx') or filename_lower.endswith('.xls'):
            df = pd.read_excel(file)
            print(f'[DEBUG] Archivo Excel leído: {len(df)} filas, {len(df.columns)} columnas')
        elif filename_lower.endswith('.csv'):
            # Intentar diferentes encodings para CSV
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            for encoding in encodings:
                try:
                    file.seek(0)  # Resetear el archivo al inicio
                    df = pd.read_csv(file, encoding=encoding)
                    print(f'[DEBUG] Archivo CSV leído con encoding {encoding}: {len(df)} filas, {len(df.columns)} columnas')
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                return jsonify(success=False, error='Error al leer el archivo CSV. Verifique el encoding del archivo.'), 400
        else:
            return jsonify(success=False, error='Formato no soportado. Use .xlsx, .xls o .csv'), 400
        
        # Verificar que el DataFrame no está vacío
        if df.empty:
            return jsonify(success=False, error='El archivo está vacío'), 400
        
    except pd.errors.EmptyDataError:
        return jsonify(success=False, error='El archivo está vacío o no contiene datos'), 400
    except pd.errors.ParserError as e:
        print(f'[ERROR] Error de parsing: {str(e)}')
        return jsonify(success=False, error=f'Error al parsear el archivo: {str(e)}'), 400
    except Exception as e:
        print(f'[ERROR] Error al leer archivo: {str(e)}')
        return jsonify(success=False, error=f'Error al leer el archivo: {str(e)}'), 400
    has_true = 'diagnosis' in df.columns
    X = prepare_df(df.drop(columns=['diagnosis'])) if has_true else prepare_df(df)
    probs = model.predict_proba(X)
    preds = [int(np.argmax(p)+1) for p in probs]
    results = []
    for i, p in enumerate(probs, start=1):
        result = {
            'row': i,
            'diagnosis': int(np.argmax(p)+1),
            'diagnosis_label': DIAGNOSIS_LABELS.get(int(np.argmax(p)+1), f'Clase {int(np.argmax(p)+1)}'),
            'probabilities': {1: float(p[0]), 2: float(p[1]), 3: float(p[2])}
        }
        if has_true:
            true_val = int(df['diagnosis'].iloc[i-1])
            result['true_diagnosis'] = true_val
            result['correct'] = result['diagnosis'] == true_val
        results.append(result)
    evaluation = None
    cm_balanced = None
    if has_true:
        y_true = df['diagnosis'].astype(int).tolist()
        acc = accuracy_score(y_true, preds)
        bacc = balanced_accuracy_score(y_true, preds)
        cm = confusion_matrix(y_true, preds, labels=[1,2,3]).tolist()
        report = classification_report(y_true, preds, output_dict=True)
        evaluation = {
            'accuracy': acc,
            'balanced_accuracy': bacc,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        # Calcular matriz de confusión en datos balanceados si SMOTE está aplicado
        model_info = TRAINING_INFO.get(model_type, {})
        if model_info.get('smote_applied'):
            # Generar datos balanceados usando SMOTE
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y_true)
            # Predecir en los datos balanceados
            preds_balanced = model.predict(X_balanced)
            # Calcular matriz de confusión balanceada
            cm_balanced = confusion_matrix(y_balanced, preds_balanced, labels=[1,2,3]).tolist()
    # Incluir información de balanceo SMOTE en la respuesta
    model_info = TRAINING_INFO.get(model_type, {})
    smote_info = None
    if model_info.get('smote_applied'):
        smote_info = {
            'applied': True,
            'original_distribution': model_info.get('original_distribution', {}),
            'balanced_distribution': model_info.get('balanced_distribution', {}),
            'original_total': model_info.get('original_total', 0),
            'balanced_total': model_info.get('balanced_total', 0),
        }
    
    return jsonify(
        success=True, 
        total_predictions=len(results), 
        results=results, 
        evaluation=evaluation, 
        model_used=model_type,
        smote_info=smote_info,
        confusion_matrix_balanced=cm_balanced
    )

if __name__ == '__main__':
    # init_models() ya se ejecutó arriba, no es necesario llamarlo de nuevo
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)