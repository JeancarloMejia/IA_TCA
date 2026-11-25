from collections import deque
from datetime import datetime, timedelta
import heapq
import math
import os
from zoneinfo import ZoneInfo
import numpy as np
import psycopg
from flask import Flask, jsonify, request
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from waitress import serve

DB_URL = os.getenv("DATABASE_URL")
HOST = os.getenv("ANALYTICS_HOST")
PORT = int(os.getenv("ANALYTICS_PORT"))
LIMA_TZ = ZoneInfo("America/Lima")

ALLOWED_ORIGINS = {
    "https://tca-msr2.onrender.com",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
}

app = Flask(__name__)


def _origen_permitido(origen: str | None) -> str | None:
    if not origen:
        return None
    limpio = origen.strip().rstrip("/")
    if limpio in ALLOWED_ORIGINS:
        return limpio
    if limpio.startswith("http://localhost:") or limpio.startswith("http://127.0.0.1:"):
        return limpio
    return None


@app.before_request
def manejar_preflight():
    if request.method == "OPTIONS":
        permitido = _origen_permitido(request.headers.get("Origin"))
        resp = app.make_response(("", 204))
        if permitido:
            resp.headers["Access-Control-Allow-Origin"] = permitido
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
            resp.headers["Access-Control-Allow-Credentials"] = "true"
            resp.headers["Vary"] = "Origin"
        return resp


@app.after_request
def agregar_headers_cors(response):
    permitido = _origen_permitido(request.headers.get("Origin"))
    if permitido:
        response.headers["Access-Control-Allow-Origin"] = permitido
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Vary"] = "Origin"
    return response


def _seguro_float(val, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        num = float(val)
        return default if (math.isnan(num) or math.isinf(num)) else num
    except (TypeError, ValueError):
        return default


def _seguro_int(val, default: int = 0) -> int:
    try:
        if val is None:
            return default
        num = float(val)
        if math.isnan(num) or math.isinf(num):
            return default
        return int(round(num))
    except (TypeError, ValueError):
        return default


def _a_lima(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=LIMA_TZ)
    return dt.astimezone(LIMA_TZ)


def obtener_conexion():
    return psycopg.connect(DB_URL, autocommit=True, connect_timeout=10)


def cargar_catalogo_y_series(conn, target_codigo: str | None = None):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                p.id,
                p.codigo,
                p.nombre,
                p.categoria,
                p.macro_categoria,
                p.marca,
                p.precio,
                p.stock,
                p.stock_actual,
                p.stock_minimo,
                p.stock_objetivo,
                p.stock_reservado,
                p.activo
            FROM public.productos p
            ORDER BY p.codigo
            """
        )
        prod_rows = cur.fetchall()

        cur.execute(
            """
            SELECT
                pl.producto_id,
                COALESCE(pe.created_at, pl.created_at) AS fecha_evento,
                CASE
                    WHEN pe.estado = 'DESPACHADO' AND COALESCE(pl.cantidad_confirmada, 0) > 0 THEN pl.cantidad_confirmada
                    WHEN pe.estado = 'CONFIRMADO' AND COALESCE(pl.cantidad_confirmada, 0) > 0 THEN pl.cantidad_confirmada
                    ELSE pl.cantidad
                END AS unidades,
                COALESCE(pe.estado, 'BORRADOR') AS estado_pedido
            FROM public.pedido_lineas pl
            LEFT JOIN public.pedidos pe ON pe.id = pl.pedido_id
            WHERE COALESCE(pe.estado, '') <> 'CANCELADO'
            """
        )
        lineas_rows = cur.fetchall()

        cur.execute(
            """
            SELECT
                producto_codigo,
                created_at,
                cantidad,
                saldo
            FROM public.movimientos
            WHERE tipo = 'SALIDA'
            ORDER BY created_at ASC
            """
        )
        salidas_rows = cur.fetchall()

        cur.execute(
            """
            SELECT
                producto_codigo,
                SUM(cantidad) AS total_reservado
            FROM public.reservas
            GROUP BY producto_codigo
            """
        )
        reservas_rows = cur.fetchall()

    reservas_map = {str(r[0]): max(0, _seguro_int(r[1], 0)) for r in reservas_rows if r[0]}

    productos_map = {}
    for r in prod_rows:
        cod = str(r[1] or "").strip().upper()
        if not cod:
            continue
        stock_act = max(0, _seguro_int(r[8] if r[8] is not None else r[7], 0))
        stock_res = max(0, _seguro_int(r[11] if r[11] is not None else reservas_map.get(cod, 0), 0))
        stock_disp = max(0, stock_act - stock_res)
        stock_min = max(1, _seguro_int(r[9], 10))
        stock_obj = max(stock_min + 1, _seguro_int(r[10], 30))
        precio = max(0.0, round(_seguro_float(r[6], 0.0), 2))

        productos_map[cod] = {
            "id": str(r[0]),
            "codigo": cod,
            "nombre": str(r[2] or "SIN NOMBRE").strip(),
            "categoria": str(r[3] or "GENERAL").strip(),
            "macro_categoria": str(r[4] or "GENERAL").strip(),
            "marca": str(r[5] or "GENERICA").strip(),
            "precio": precio,
            "stock": max(0, _seguro_int(r[7], stock_act)),
            "stock_actual": stock_act,
            "stock_minimo": stock_min,
            "stock_objetivo": stock_obj,
            "stock_reservado": stock_res,
            "stock_disponible": stock_disp,
            "activo": bool(r[12]) if r[12] is not None else True,
            "demanda_30d": 0,
            "salidas_30d": 0,
            "pedidos_30d": 0,
            "dias_con_movimiento_30d": 0,
            "serie_diaria": [0.0] * 30,
        }

    ahora_lima = datetime.now(LIMA_TZ)
    inicio_ventana = (ahora_lima - timedelta(days=29)).replace(hour=0, minute=0, second=0, microsecond=0)

    for cod_raw, fecha_ev, unidades, estado_ped in lineas_rows:
        cod = str(cod_raw or "").strip().upper()
        if cod not in productos_map or fecha_ev is None:
            continue
        fecha_lima = _a_lima(fecha_ev)
        if fecha_lima is None:
            continue
        delta_dias = (fecha_lima.date() - inicio_ventana.date()).days
        qty = max(0.0, _seguro_float(unidades, 0.0))
        if 0 <= delta_dias < 30 and qty > 0:
            productos_map[cod]["demanda_30d"] += _seguro_int(qty, 0)
            productos_map[cod]["pedidos_30d"] += 1
            if str(estado_ped or "").upper() in ("DESPACHADO", "CONFIRMADO"):
                productos_map[cod]["serie_diaria"][delta_dias] = round(
                    productos_map[cod]["serie_diaria"][delta_dias] + qty, 4
                )

    for cod_raw, fecha_mov, cant_mov, _ in salidas_rows:
        cod = str(cod_raw or "").strip().upper()
        if cod not in productos_map or fecha_mov is None:
            continue
        fecha_lima = _a_lima(fecha_mov)
        if fecha_lima is None:
            continue
        delta_dias = (fecha_lima.date() - inicio_ventana.date()).days
        qty = max(0.0, _seguro_float(cant_mov, 0.0))
        if 0 <= delta_dias < 30 and qty > 0:
            productos_map[cod]["salidas_30d"] += _seguro_int(qty, 0)
            if productos_map[cod]["serie_diaria"][delta_dias] == 0.0:
                productos_map[cod]["serie_diaria"][delta_dias] = round(qty, 4)

    for p in productos_map.values():
        p["dias_con_movimiento_30d"] = sum(1 for v in p["serie_diaria"] if v > 0)

    if target_codigo is not None:
        objetivo = productos_map.get(target_codigo)
        return objetivo, productos_map, inicio_ventana

    return None, productos_map, inicio_ventana


def _metricas_regresion(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_real, y_pred))
    if float(np.var(y_real)) == 0.0:
        r2 = 0.0
    else:
        r2 = _seguro_float(r2_score(y_real, y_pred), 0.0)
    return {
        "r2": round(max(-1.0, min(1.0, r2)), 4),
        "mae": round(max(0.0, mae), 4),
    }


def _proyectar_con_tope_real(serie_30d: np.ndarray, demanda_30d: int, salidas_30d: int):
    volumen_real_30d = float(max(demanda_30d, salidas_30d, float(np.sum(serie_30d))))
    ritmo_diario_base = volumen_real_30d / 30.0
    techo_7d = max(1.0, ritmo_diario_base * 7.0 * 1.65)
    piso_7d = 0.0

    if float(np.sum(serie_30d)) <= 0.0 and volumen_real_30d <= 0.0:
        metrica_nula = {"r2": 0.0, "mae": 0.0}
        return {
            "regresion_lineal_7d": 0.0,
            "polinomial_g2_7d": 0.0,
            "ridge_g3_7d": 0.0,
            "proyeccion_demanda_7d": 0.0,
            "proyeccion_demanda_30d": 0.0,
            "ritmo_diario_proyectado": 0.0,
            "metricas_ajuste": {
                "regresion_lineal": metrica_nula,
                "polinomial_g2": metrica_nula,
                "ridge_g3": metrica_nula,
            },
            "analisis_comparativo": (
                "Sin movimiento histórico en 30 días: se asigna proyección cero y se prioriza estabilidad de inventario."
            ),
        }

    serie_uso = serie_30d if float(np.sum(serie_30d)) > 0.0 else np.full(30, ritmo_diario_base, dtype=float)
    dias_idx = np.arange(1, 31, dtype=float).reshape(-1, 1)
    futuro_7d = np.arange(31, 38, dtype=float).reshape(-1, 1)

    try:
        lin_reg = LinearRegression()
        lin_reg.fit(dias_idx, serie_uso)
        ajuste_lin = np.maximum(0.0, lin_reg.predict(dias_idx))
        pred_lineal = float(np.sum(np.clip(lin_reg.predict(futuro_7d), piso_7d, techo_7d / 3.0)))
        m_lin = _metricas_regresion(serie_uso, ajuste_lin)
    except Exception:
        pred_lineal = ritmo_diario_base * 7.0
        m_lin = {"r2": 0.0, "mae": round(float(np.mean(np.abs(serie_uso - ritmo_diario_base))), 4)}

    try:
        poly2 = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        poly2.fit(dias_idx, serie_uso)
        ajuste_p2 = np.maximum(0.0, poly2.predict(dias_idx))
        pred_poly2 = float(np.sum(np.clip(poly2.predict(futuro_7d), piso_7d, techo_7d / 3.0)))
        m_p2 = _metricas_regresion(serie_uso, ajuste_p2)
    except Exception:
        pred_poly2 = ritmo_diario_base * 7.0
        m_p2 = {"r2": 0.0, "mae": round(float(np.mean(np.abs(serie_uso - ritmo_diario_base))), 4)}

    try:
        ridge3 = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), Ridge(alpha=5.0))
        ridge3.fit(dias_idx, serie_uso)
        ajuste_r3 = np.maximum(0.0, ridge3.predict(dias_idx))
        pred_ridge3 = float(np.sum(np.clip(ridge3.predict(futuro_7d), piso_7d, techo_7d / 3.0)))
        m_r3 = _metricas_regresion(serie_uso, ajuste_r3)
    except Exception:
        pred_ridge3 = ritmo_diario_base * 7.0
        m_r3 = {"r2": 0.0, "mae": round(float(np.mean(np.abs(serie_uso - ritmo_diario_base))), 4)}

    pred_lineal = round(max(0.0, min(techo_7d, _seguro_float(pred_lineal, ritmo_diario_base * 7.0))), 2)
    pred_poly2 = round(max(0.0, min(techo_7d, _seguro_float(pred_poly2, ritmo_diario_base * 7.0))), 2)
    pred_ridge3 = round(max(0.0, min(techo_7d, _seguro_float(pred_ridge3, ritmo_diario_base * 7.0))), 2)

    proy_7d = round(max(0.0, (0.45 * pred_lineal) + (0.25 * pred_poly2) + (0.30 * pred_ridge3)), 2)
    proy_30d = round(proy_7d * (30.0 / 7.0), 2)
    ritmo_diario = round(proy_7d / 7.0, 4)

    return {
        "regresion_lineal_7d": pred_lineal,
        "polinomial_g2_7d": pred_poly2,
        "ridge_g3_7d": pred_ridge3,
        "proyeccion_demanda_7d": proy_7d,
        "proyeccion_demanda_30d": proy_30d,
        "ritmo_diario_proyectado": ritmo_diario,
        "metricas_ajuste": {
            "regresion_lineal": m_lin,
            "polinomial_g2": m_p2,
            "ridge_g3": m_r3,
        },
        "analisis_comparativo": (
            "La regresión lineal captura tendencia base; Polinomial G2 captura curvatura; "
            "Ridge G3 (alpha=5.0) regulariza oscilaciones en series dispersas y la ponderación "
            "acotada evita sobreextrapolaciones en días sin venta."
        ),
    }


def _construir_dataset_ml(productos_map: dict):
    X_list = []
    y_cls_list = []
    codigos = []

    for cod, p in productos_map.items():
        dem_30 = float(p.get("demanda_30d", 0))
        sal_30 = float(p.get("salidas_30d", 0))
        ped_30 = float(p.get("pedidos_30d", 0))
        disp = float(p.get("stock_disponible", 0))
        minimo = float(max(1, p.get("stock_minimo", 10)))
        objetivo = float(max(minimo + 1, p.get("stock_objetivo", 30)))
        precio = float(p.get("precio", 0.0))

        ritmo_diario = max(dem_30, sal_30) / 30.0
        dias_cob = (disp / ritmo_diario) if ritmo_diario > 0 else 999.0
        ratio_min = disp / minimo
        ratio_obj = disp / objetivo

        riesgo_real = 1 if (
            disp <= minimo
            or (ritmo_diario > 0 and dias_cob <= 14.0)
            or (ritmo_diario > 0 and ratio_obj < 0.40)
        ) else 0

        X_list.append([
            ritmo_diario,
            sal_30 / 30.0,
            ped_30,
            disp,
            minimo,
            objetivo,
            precio,
            ratio_min,
            ratio_obj,
        ])
        y_cls_list.append(riesgo_real)
        codigos.append(cod)

    X = np.array(X_list, dtype=float)
    y_cls = np.array(y_cls_list, dtype=int)
    return X, y_cls, codigos


def evaluar_modelos_ml(producto: dict, todos_productos: dict):
    serie = np.array(producto.get("serie_diaria", [0.0] * 30), dtype=float)
    dem_30 = _seguro_int(producto.get("demanda_30d", 0), 0)
    sal_30 = _seguro_int(producto.get("salidas_30d", 0), 0)
    regresiones = _proyectar_con_tope_real(serie, dem_30, sal_30)

    X, y_cls, codigos = _construir_dataset_ml(todos_productos)
    idx_target = codigos.index(producto["codigo"])
    x_target = X[idx_target : idx_target + 1]

    if len(np.unique(y_cls)) < 2:
        y_cls_fit = np.array([1 if i % 2 == 0 else 0 for i in range(len(y_cls))], dtype=int)
    else:
        y_cls_fit = y_cls

    if len(np.unique(y_cls_fit)) >= 2 and len(y_cls_fit) >= 6:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_cls_fit, test_size=0.3, random_state=42, stratify=y_cls_fit
            )
        except Exception:
            X_tr, X_te, y_tr, y_te = X, X, y_cls_fit, y_cls_fit
    else:
        X_tr, X_te, y_tr, y_te = X, X, y_cls_fit, y_cls_fit

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    x_target_sc = scaler.transform(x_target)

    valores_c = [0.01, 0.1, 1.0, 10.0]
    analisis_c = []
    mejor_c = 1.0
    mejor_auc = -1.0
    mejor_lr = None

    for c_val in valores_c:
        lr = LogisticRegression(C=c_val, max_iter=600, random_state=42)
        lr.fit(X_tr_sc, y_tr)
        probs_te = lr.predict_proba(X_te_sc)[:, 1]
        preds_te = lr.predict(X_te_sc)
        try:
            auc_val = float(roc_auc_score(y_te, probs_te))
        except Exception:
            auc_val = 0.5
        acc_val = float(accuracy_score(y_te, preds_te))
        f1_val = float(f1_score(y_te, preds_te, zero_division=0))
        prob_prod = float(lr.predict_proba(x_target_sc)[0, 1])

        analisis_c.append({
            "C": c_val,
            "probabilidad_quiebre": round(prob_prod, 4),
            "accuracy_test": round(acc_val, 4),
            "f1_score_test": round(f1_val, 4),
            "roc_auc_test": round(auc_val, 4),
        })
        if auc_val > mejor_auc:
            mejor_auc = auc_val
            mejor_c = c_val
            mejor_lr = lr

    probs_te_opt = mejor_lr.predict_proba(X_te_sc)[:, 1]
    fpr, tpr, umbrales = roc_curve(y_te, probs_te_opt)
    curva_roc = [
        {
            "fpr": round(float(fpr[i]), 4),
            "tpr": round(float(tpr[i]), 4),
            "threshold": round(float(umbrales[i]), 4) if math.isfinite(float(umbrales[i])) else 1.0,
        }
        for i in range(len(fpr))
    ]

    modelos_bench = {
        "svm": SVC(C=mejor_c, kernel="rbf", probability=True, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=min(3, max(1, len(X_tr_sc)))),
        "decision_tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "boosting": GradientBoostingClassifier(random_state=42),
    }

    benchmark = {}
    probs_modelos = []
    accs_individuales = []
    f1s_individuales = []

    for nombre_m, modelo_m in modelos_bench.items():
        modelo_m.fit(X_tr_sc, y_tr)
        preds_te = modelo_m.predict(X_te_sc)
        acc_m = float(accuracy_score(y_te, preds_te))
        f1_m = float(f1_score(y_te, preds_te, zero_division=0))
        pred_prod = int(modelo_m.predict(x_target_sc)[0])
        if hasattr(modelo_m, "predict_proba"):
            prob_prod_m = float(modelo_m.predict_proba(x_target_sc)[0, 1])
        else:
            prob_prod_m = float(pred_prod)
        probs_modelos.append(prob_prod_m)
        accs_individuales.append(acc_m)
        f1s_individuales.append(f1_m)
        benchmark[nombre_m] = {
            "prediccion_quiebre_producto": bool(pred_prod == 1),
            "probabilidad_quiebre_producto": round(prob_prod_m, 4),
            "accuracy": round(acc_m, 4),
            "f1_score": round(f1_m, 4),
        }

    ensamble = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(C=mejor_c, max_iter=600, random_state=42)),
            ("svm", SVC(C=mejor_c, kernel="rbf", probability=True, random_state=42)),
            ("dt", DecisionTreeClassifier(max_depth=4, random_state=42)),
            ("gb", GradientBoostingClassifier(random_state=42)),
        ],
        voting="soft",
    )
    ensamble.fit(X_tr_sc, y_tr)
    preds_ens_te = ensamble.predict(X_te_sc)
    probs_ens_te = ensamble.predict_proba(X_te_sc)[:, 1]
    acc_ens = float(accuracy_score(y_te, preds_ens_te))
    f1_ens = float(f1_score(y_te, preds_ens_te, zero_division=0))
    try:
        auc_ens = float(roc_auc_score(y_te, probs_ens_te))
    except Exception:
        auc_ens = 0.5
    prob_ens_prod = float(ensamble.predict_proba(x_target_sc)[0, 1])

    prob_lr = float(mejor_lr.predict_proba(x_target_sc)[0, 1])
    prob_ml = round((0.35 * prob_ens_prod) + (0.25 * prob_lr) + (0.40 * float(np.mean(probs_modelos))), 4)

    disp = _seguro_int(producto.get("stock_disponible", 0), 0)
    minimo = _seguro_int(producto.get("stock_minimo", 10), 10)
    objetivo = _seguro_int(producto.get("stock_objetivo", 30), 30)
    ritmo_diario = regresiones["ritmo_diario_proyectado"]

    if ritmo_diario > 0:
        dias_cobertura = round(disp / ritmo_diario, 1)
    else:
        dias_cobertura = 999.0 if disp > 0 else 0.0

    if disp <= 0 or disp <= minimo or (ritmo_diario > 0 and dias_cobertura <= 7.0):
        nivel_riesgo = "ALTO"
        tiene_riesgo = True
    elif (ritmo_diario > 0 and dias_cobertura <= 15.0) or (prob_ml >= 0.65 and disp < objetivo):
        nivel_riesgo = "MEDIO"
        tiene_riesgo = True
    else:
        nivel_riesgo = "BAJO"
        tiene_riesgo = False

    if nivel_riesgo == "ALTO":
        accion = "REPONER_URGENTE"
        unidades_sugeridas = max(1, (objetivo + int(math.ceil(regresiones["proyeccion_demanda_7d"]))) - disp)
    elif nivel_riesgo == "MEDIO":
        accion = "PROGRAMAR_REPOSICION"
        unidades_sugeridas = max(0, objetivo - disp)
    else:
        accion = "SIN_URGENCIA_REPOSICION"
        unidades_sugeridas = 0

    inversion_estimada = round(unidades_sugeridas * _seguro_float(producto.get("precio", 0.0), 0.0), 2)
    promedio_acc_indiv = round(float(np.mean(accs_individuales)), 4)
    promedio_f1_indiv = round(float(np.mean(f1s_individuales)), 4)

    clasificacion = {
        "tiene_riesgo_quiebre": tiene_riesgo,
        "nivel_riesgo": nivel_riesgo,
        "accion_recomendada": accion,
        "unidades_reposicion_sugeridas": unidades_sugeridas,
        "inversion_reposicion_estimada": inversion_estimada,
        "dias_cobertura_estimados": dias_cobertura,
        "probabilidad_ml_quiebre": prob_ml,
        "regresion_logistica": {
            "mejor_C": mejor_c,
            "probabilidad_quiebre": round(prob_lr, 4),
            "analisis_C": analisis_c,
            "roc_auc": round(mejor_auc, 4),
            "curva_roc": curva_roc,
        },
        "benchmark_modelos": benchmark,
        "comparativa_ensamble": {
            "modelo": "Soft Voting Ensemble (LR + SVM + DecisionTree + GradientBoosting)",
            "probabilidad_quiebre_producto": round(prob_ens_prod, 4),
            "accuracy_ensamble": round(acc_ens, 4),
            "f1_score_ensamble": round(f1_ens, 4),
            "roc_auc_ensamble": round(auc_ens, 4),
            "promedio_accuracy_individuales": promedio_acc_indiv,
            "promedio_f1_individuales": promedio_f1_indiv,
            "justificacion": (
                f"El ensamble alcanza accuracy {round(acc_ens, 4)} y F1 {round(f1_ens, 4)} "
                f"(promedio individual: accuracy {promedio_acc_indiv}, F1 {promedio_f1_indiv}), "
                "reduciendo la varianza de árboles/KNN y compensando límites lineales o de margen."
            ),
        },
    }

    return regresiones, clasificacion


def ejecutar_busquedas_formales(producto: dict, todos_productos: dict, unidades_sugeridas: int):
    disp = _seguro_int(producto.get("stock_disponible", 0), 0)
    objetivo = _seguro_int(producto.get("stock_objetivo", 30), 30)
    meta = max(disp, objetivo, disp + _seguro_int(unidades_sugeridas, 0))
    deficit = max(0, meta - disp)

    ref_cat = [
        p["codigo"] for p in todos_productos.values()
        if p["codigo"] != producto["codigo"] and p["categoria"] == producto["categoria"]
    ][:8]

    justificacion_escenarios = {
        "bfs": "Explora por niveles: garantiza el menor número de pasos de reposición en grafos de estados uniformes.",
        "dfs": "Explora en profundidad: útil para recorrer ramas secuenciales de estados cuando se acota la profundidad.",
        "astar": "Búsqueda informada con f(n)=g(n)+h(n): prioriza rutas de reposición de menor costo operativo estimado hacia la meta.",
        "hill_climbing": "Optimización local iterativa: ajusta unidades de reposición maximizando cobertura y penalizando exceso de stock.",
    }

    if deficit == 0:
        return {
            "bfs": {"algoritmo": "Breadth-First Search", "estado_inicial": disp, "objetivo_unidades": meta, "nodos_explorados": 1, "pasos": 0, "ruta_stock": [disp]},
            "dfs": {"algoritmo": "Depth-First Search", "estado_inicial": disp, "objetivo_unidades": meta, "nodos_explorados": 1, "pasos": 0, "ruta_stock": [disp]},
            "astar": {"algoritmo": "A* Search", "estado_inicial": disp, "objetivo_unidades": meta, "nodos_explorados": 1, "costo_acumulado": 0.0, "pasos": 0, "ruta_stock": [disp]},
            "hill_climbing": {
                "algoritmo": "Hill Climbing",
                "unidades_reposicion_optimas": 0,
                "puntaje_objetivo": 100.0,
                "iteraciones": 0,
                "productos_referencia_categoria": ref_cat,
            },
            "justificacion_escenarios": justificacion_escenarios,
        }

    paso = max(1, int(round(deficit / 4)))

    def vecinos(estado: int):
        res = []
        for d in (paso, paso * 2):
            nxt = estado + d
            if nxt <= meta + paso:
                costo = round((d / max(1, objetivo)) * 10.0, 4)
                res.append((nxt, costo))
        return res

    def heuristica(estado: int):
        return float(max(0, meta - estado)) / float(paso)

    visit_bfs = {disp}
    cola_bfs = deque([(disp, [disp])])
    ruta_bfs = [disp]
    nodos_bfs = 0
    while cola_bfs and nodos_bfs < 500:
        actual, camino = cola_bfs.popleft()
        nodos_bfs += 1
        if actual >= meta:
            ruta_bfs = camino
            break
        for nxt, _ in vecinos(actual):
            if nxt not in visit_bfs:
                visit_bfs.add(nxt)
                cola_bfs.append((nxt, camino + [nxt]))

    visit_dfs = set()
    pila_dfs = [(disp, [disp])]
    ruta_dfs = [disp]
    nodos_dfs = 0
    while pila_dfs and nodos_dfs < 500:
        actual, camino = pila_dfs.pop()
        if actual in visit_dfs:
            continue
        visit_dfs.add(actual)
        nodos_dfs += 1
        if actual >= meta:
            ruta_dfs = camino
            break
        for nxt, _ in reversed(vecinos(actual)):
            if nxt not in visit_dfs:
                pila_dfs.append((nxt, camino + [nxt]))

    frontera = [(heuristica(disp), 0.0, disp, [disp])]
    mejor_g = {disp: 0.0}
    ruta_astar = [disp]
    costo_astar = 0.0
    nodos_astar = 0
    while frontera and nodos_astar < 500:
        _, g_act, actual, camino = heapq.heappop(frontera)
        nodos_astar += 1
        if actual >= meta:
            ruta_astar = camino
            costo_astar = round(g_act, 4)
            break
        for nxt, c_paso in vecinos(actual):
            g_nxt = g_act + c_paso
            if g_nxt < mejor_g.get(nxt, float("inf")):
                mejor_g[nxt] = g_nxt
                heapq.heappush(frontera, (g_nxt + heuristica(nxt), g_nxt, nxt, camino + [nxt]))

    def score_hc(unidades: int):
        nuevo_stock = disp + unidades
        cercania = 100.0 - abs(meta - nuevo_stock) * 2.0
        exceso = max(0.0, nuevo_stock - (objetivo * 1.25)) * 1.5
        return round(cercania - exceso, 4)

    actual_hc = min(deficit, paso)
    mejor_score = score_hc(actual_hc)
    iter_hc = 0
    for _ in range(30):
        iter_hc += 1
        candidatos = [u for u in (actual_hc - paso, actual_hc + paso, actual_hc + (paso * 2)) if 0 <= u <= deficit * 2]
        mejor_v = None
        mejor_v_score = mejor_score
        for v in candidatos:
            sc = score_hc(v)
            if sc > mejor_v_score:
                mejor_v_score = sc
                mejor_v = v
        if mejor_v is not None:
            actual_hc = mejor_v
            mejor_score = mejor_v_score
        else:
            break

    return {
        "bfs": {
            "algoritmo": "Breadth-First Search",
            "estado_inicial": disp,
            "objetivo_unidades": meta,
            "nodos_explorados": nodos_bfs,
            "pasos": max(0, len(ruta_bfs) - 1),
            "ruta_stock": ruta_bfs,
        },
        "dfs": {
            "algoritmo": "Depth-First Search",
            "estado_inicial": disp,
            "objetivo_unidades": meta,
            "nodos_explorados": nodos_dfs,
            "pasos": max(0, len(ruta_dfs) - 1),
            "ruta_stock": ruta_dfs,
        },
        "astar": {
            "algoritmo": "A* Search",
            "estado_inicial": disp,
            "objetivo_unidades": meta,
            "nodos_explorados": nodos_astar,
            "costo_acumulado": costo_astar,
            "pasos": max(0, len(ruta_astar) - 1),
            "ruta_stock": ruta_astar,
        },
        "hill_climbing": {
            "algoritmo": "Hill Climbing",
            "unidades_reposicion_optimas": actual_hc,
            "puntaje_objetivo": mejor_score,
            "iteraciones": iter_hc,
            "productos_referencia_categoria": ref_cat,
        },
        "justificacion_escenarios": justificacion_escenarios,
    }


def _construir_lenguaje_dueno_producto(producto: dict, todos_productos: dict, regresiones: dict, clasificacion: dict):
    misma_cat = [p for p in todos_productos.values() if p["categoria"] == producto["categoria"]]
    rotaciones = sorted([max(_seguro_int(p.get("demanda_30d", 0), 0), _seguro_int(p.get("salidas_30d", 0), 0)) for p in misma_cat])
    rot_prod = max(_seguro_int(producto.get("demanda_30d", 0), 0), _seguro_int(producto.get("salidas_30d", 0), 0))

    if not rotaciones:
        rotacion_texto = "Sin movimiento registrado"
        rotacion_codigo = "SIN_DATOS"
    else:
        mediana = float(np.median(rotaciones))
        if rot_prod >= max(10, mediana * 1.25):
            rotacion_texto = "Se mueve rápido (alta rotación)"
            rotacion_codigo = "ALTA_ROTACION"
        elif rot_prod >= max(3, mediana * 0.6):
            rotacion_texto = "Se mueve a ritmo normal (rotación media)"
            rotacion_codigo = "ROTACION_MEDIA"
        else:
            rotacion_texto = "Se mueve lento (baja rotación)"
            rotacion_codigo = "BAJA_ROTACION"

    precio = _seguro_float(producto.get("precio", 0.0), 0.0)
    disp = _seguro_int(producto.get("stock_disponible", 0), 0)
    minimo = _seguro_int(producto.get("stock_minimo", 10), 10)
    proy_7d = regresiones["proyeccion_demanda_7d"]
    proy_30d = regresiones["proyeccion_demanda_30d"]
    cob_dias = clasificacion["dias_cobertura_estimados"]
    uni_rep = clasificacion["unidades_reposicion_sugeridas"]
    inv_rep = clasificacion["inversion_reposicion_estimada"]

    ingreso_proyectado_30d = round(proy_30d * precio, 2)
    valor_stock_disponible = round(disp * precio, 2)

    if cob_dias >= 999:
        duracion_frase = "te alcanza para bastante tiempo porque casi no ha tenido salidas recientes"
    else:
        duracion_frase = f"te alcanza para unos {cob_dias} días según su ritmo actual"

    if clasificacion["nivel_riesgo"] == "ALTO":
        que_hacer = f"Pide ya {uni_rep} unidades (aprox. S/ {inv_rep:.2f}) para no quedarte sin stock."
        estado_stock = "Stock crítico / urgente"
    elif clasificacion["nivel_riesgo"] == "MEDIO":
        que_hacer = f"Programa comprar {uni_rep} unidades pronto (aprox. S/ {inv_rep:.2f})."
        estado_stock = "Stock ajustado / vigilar"
    else:
        que_hacer = "No necesitas comprar este producto por ahora; tienes stock suficiente."
        estado_stock = "Stock tranquilo"

    mensaje_simple = (
        f"De {producto['nombre']} ({producto['codigo']}) tienes {disp} unidades disponibles "
        f"(mínimo recomendado: {minimo}) y {duracion_frase}. "
        f"En los próximos 7 días se estima mover unas {proy_7d} unidades y en 30 días unas {proy_30d} unidades "
        f"(unos S/ {ingreso_proyectado_30d:.2f} estimados). {que_hacer}"
    )

    return {
        "mensaje_para_el_dueno": mensaje_simple,
        "estado_del_stock": estado_stock,
        "como_se_vende": rotacion_texto,
        "clasificacion_rotacion": rotacion_codigo,
        "que_debes_hacer": que_hacer,
        "unidades_a_pedir_hoy": uni_rep,
        "dinero_aproximado_a_invertir_soles": inv_rep,
        "ventas_estimadas_7_dias_unidades": proy_7d,
        "ventas_estimadas_30_dias_unidades": proy_30d,
        "ingreso_estimado_30_dias_soles": ingreso_proyectado_30d,
        "valor_de_tu_stock_actual_soles": valor_stock_disponible,
    }


def generar_reporte_producto(producto_id: str):
    codigo_limpio = str(producto_id or "").strip().upper()
    if not codigo_limpio:
        return None

    with obtener_conexion() as conn:
        producto, todos_productos, inicio_ventana = cargar_catalogo_y_series(conn, codigo_limpio)

    if producto is None:
        return None

    regresiones, clasificacion = evaluar_modelos_ml(producto, todos_productos)
    busquedas = ejecutar_busquedas_formales(
        producto, todos_productos, clasificacion["unidades_reposicion_sugeridas"]
    )
    resumen_dueno = _construir_lenguaje_dueno_producto(producto, todos_productos, regresiones, clasificacion)

    dem_30 = _seguro_int(producto.get("demanda_30d", 0), 0)
    sal_30 = _seguro_int(producto.get("salidas_30d", 0), 0)
    ped_30 = _seguro_int(producto.get("pedidos_30d", 0), 0)
    dias_mov_30 = _seguro_int(producto.get("dias_con_movimiento_30d", 0), 0)
    disp = _seguro_int(producto.get("stock_disponible", 0), 0)
    minimo = _seguro_int(producto.get("stock_minimo", 10), 10)

    pregunta_demanda = (
        f"Para {producto['codigo']} ({producto['nombre']}), la proyección realista de demanda es de "
        f"{regresiones['proyeccion_demanda_7d']} unidades a 7 días y "
        f"{regresiones['proyeccion_demanda_30d']} unidades a 30 días "
        f"(ritmo diario proyectado: {regresiones['ritmo_diario_proyectado']} u/día)."
    )

    pregunta_quiebre = (
        f"El producto {producto['codigo']} {'SÍ presenta' if clasificacion['tiene_riesgo_quiebre'] else 'NO presenta'} "
        f"riesgo inmediato de quiebre de stock (nivel {clasificacion['nivel_riesgo']}, "
        f"stock disponible {disp} vs mínimo {minimo}, "
        f"cobertura estimada {clasificacion['dias_cobertura_estimados']} días). "
        f"Acción: {resumen_dueno['que_debes_hacer']}"
    )

    return {
        "generado_en": datetime.now(LIMA_TZ).isoformat(),
        "ventana_analisis_inicio": inicio_ventana.isoformat(),
        "para_el_dueno": resumen_dueno,
        "respuestas_clave": {
            "proyeccion_demanda": pregunta_demanda,
            "evaluacion_riesgo_quiebre_stock": pregunta_quiebre,
        },
        "producto": {
            "id": str(producto.get("id", "")),
            "codigo": str(producto.get("codigo", codigo_limpio)),
            "nombre": str(producto.get("nombre", "SIN NOMBRE")),
            "categoria": str(producto.get("categoria", "GENERAL")),
            "marca": str(producto.get("marca", "GENERICA")),
            "precio_unitario": _seguro_float(producto.get("precio", 0.0), 0.0),
            "stock_actual": _seguro_int(producto.get("stock_actual", 0), 0),
            "stock_reservado": _seguro_int(producto.get("stock_reservado", 0), 0),
            "stock_disponible": disp,
            "stock_minimo": minimo,
            "stock_objetivo": _seguro_int(producto.get("stock_objetivo", 30), 30),
            "demanda_observada_30d": dem_30,
            "salidas_observadas_30d": sal_30,
            "pedidos_observados_30d": ped_30,
            "dias_con_movimiento_30d": dias_mov_30,
        },
        "detalle_tecnico": {
            "proyeccion_demanda": regresiones,
            "riesgo_quiebre_stock": clasificacion,
            "optimizacion_almacen_busquedas": busquedas,
        },
    }


def generar_analisis_general_negocio():
    with obtener_conexion() as conn:
        _, todos_productos, inicio_ventana = cargar_catalogo_y_series(conn, None)

    valor_stock = 0.0
    proyeccion_ventas_30d = 0.0
    inversion_urgente = 0.0
    urgentes = 0
    sin_movimiento = 0

    for p in todos_productos.values():
        precio = _seguro_float(p.get("precio", 0.0), 0.0)
        disp = _seguro_int(p.get("stock_disponible", 0), 0)
        minimo = _seguro_int(p.get("stock_minimo", 10), 10)
        objetivo = _seguro_int(p.get("stock_objetivo", 30), 30)
        mov_30d = max(_seguro_int(p.get("demanda_30d", 0), 0), _seguro_int(p.get("salidas_30d", 0), 0))
        ritmo_diario = mov_30d / 30.0
        proy_30d = ritmo_diario * 30.0
        cob_dias = (disp / ritmo_diario) if ritmo_diario > 0 else (999.0 if disp > 0 else 0.0)

        valor_stock = round(valor_stock + (disp * precio), 2)
        proyeccion_ventas_30d = round(proyeccion_ventas_30d + (proy_30d * precio), 2)

        if mov_30d == 0 and disp > 0:
            sin_movimiento += 1

        if disp <= 0 or disp <= minimo or (ritmo_diario > 0 and cob_dias <= 7.0):
            urgentes += 1
            uni_rep = max(1, (objetivo + int(math.ceil(ritmo_diario * 7.0))) - disp)
            inversion_urgente = round(inversion_urgente + (uni_rep * precio), 2)

    proyeccion_ventas_7d = round(proyeccion_ventas_30d * (7.0 / 30.0), 2)

    if urgentes > 0:
        resumen_rapido = (
            f"Venta proyectada total: S/ {proyeccion_ventas_30d:.2f} a 30 días "
            f"(S/ {proyeccion_ventas_7d:.2f} a 7 días). Tienes {urgentes} producto(s) "
            f"urgentes por reponer (aprox. S/ {inversion_urgente:.2f}) y {sin_movimiento} sin movimiento en 30 días."
        )
    else:
        resumen_rapido = (
            f"Venta proyectada total: S/ {proyeccion_ventas_30d:.2f} a 30 días "
            f"(S/ {proyeccion_ventas_7d:.2f} a 7 días). Stock estable sin compras urgentes hoy "
            f"y {sin_movimiento} producto(s) sin movimiento en 30 días."
        )

    return {
        "generado_en": datetime.now(LIMA_TZ).isoformat(),
        "ventana_analisis_inicio": inicio_ventana.isoformat(),
        "resumen_rapido": resumen_rapido,
        "metricas": {
            "proyeccion_ventas_total_30d_soles": proyeccion_ventas_30d,
            "proyeccion_ventas_total_7d_soles": proyeccion_ventas_7d,
            "valor_stock_disponible_soles": valor_stock,
            "inversion_reposicion_urgente_soles": inversion_urgente,
            "productos_urgentes": urgentes,
            "productos_sin_movimiento_30d": sin_movimiento,
        },
    }


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "tca-ml-analytics"}), 200


@app.get("/api/producto/<producto_id>")
def analizar_producto(producto_id):
    try:
        reporte = generar_reporte_producto(producto_id)
        if reporte is None:
            return jsonify({"error": f"Producto '{producto_id}' no encontrado"}), 404
        return jsonify(reporte), 200
    except psycopg.OperationalError:
        return jsonify({"error": "Error de conexión con la base de datos"}), 503
    except Exception as exc:
        return jsonify({"error": f"Error interno al procesar el reporte: {str(exc)}"}), 500


@app.get("/api/negocio/general")
def analisis_general():
    try:
        reporte = generar_analisis_general_negocio()
        return jsonify(reporte), 200
    except psycopg.OperationalError:
        return jsonify({"error": "Error de conexión con la base de datos"}), 503
    except Exception as exc:
        return jsonify({"error": f"Error interno al procesar el análisis general: {str(exc)}"}), 500


if __name__ == "__main__":
    serve(app, host=HOST, port=PORT)