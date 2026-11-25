import os
import random
import uuid
from datetime import datetime, timedelta, timezone
import psycopg

DB_URL = os.getenv("DATABASE_URL")

NOMBRES = [
    "Carlos", "Luis", "Jose", "Jorge", "Miguel", "Diego", "Andres", "Fernando",
    "Ricardo", "Martin", "Daniel", "Javier", "Eduardo", "Raul", "Hector",
    "Maria", "Ana", "Lucia", "Sofia", "Valeria", "Camila", "Paola", "Carla",
    "Patricia", "Rosa", "Elena", "Andrea", "Fiorella", "Milagros", "Melissa",
]

APELLIDOS = [
    "Mendoza", "Quispe", "Rojas", "Huaman", "Chavez", "Castillo", "Vargas",
    "Flores", "Sánchez", "Ramos", "Torres", "Gutiérrez", "Paredes", "Silva",
    "Salazar", "Castro", "Romero", "Herrera", "Guzmán", "Ponce", "Núñez",
    "Espinoza", "Cabrera", "Delgado", "Ortega", "Vega", "Peña", "Campos",
]

DIRECCIONES = [
    "Av. Universitaria 1420, Comas, Lima",
    "Jr. Trujillo 518, Rimac, Lima",
    "Av. Próceres de la Independencia 2890, San Juan de Lurigancho, Lima",
    "Calle Los Laureles 345, Los Olivos, Lima",
    "Av. Tupac Amaru 4610, Independencia, Lima",
    "Jr. Amazonas 210, Cercado de Lima, Lima",
    "Av. Alfredo Mendiola 3540, San Martin de Porres, Lima",
    "Calle Las Gardenias 189, Puente Piedra, Lima",
    "Av. Colonial 1980, Callao",
    "Jr. Huallaga 640, Lima",
    "Av. Abancay 725, Cercado de Lima, Lima",
    "Calle Santa Rosa 412, Carabayllo, Lima",
    "Av. Gerardo Unger 5110, Los Olivos, Lima",
    "Jr. Junín 388, Barrios Altos, Lima",
    "Av. Grau 1120, Piura",
    "Calle San Martín 490, Castilla, Piura",
    "Av. Loreto 830, Piura",
    "Jr. Ayacucho 275, Sullana, Piura",
    "Av. Sánchez Cerro 1540, Piura",
    "Calle Lima 615, Chiclayo, Lambayeque",
]


def generar_clientes(rng: random.Random, total: int = 35) -> list[dict]:
    clientes = []
    documentos_usados = set()
    for _ in range(total):
        while True:
            doc = f"{rng.randint(10000000, 79999999)}"
            if doc not in documentos_usados:
                documentos_usados.add(doc)
                break
        nombre_completo = f"{rng.choice(NOMBRES)} {rng.choice(APELLIDOS)} {rng.choice(APELLIDOS)}"
        telefono = f"9{rng.randint(10000000, 99999999)}"
        direccion = rng.choice(DIRECCIONES)
        clientes.append(
            {
                "id": str(uuid.uuid4()),
                "documento": doc,
                "nombre": nombre_completo,
                "telefono": telefono,
                "direccion": direccion,
            }
        )
    return clientes


def poblar_base_datos(num_pedidos: int = 220, num_clientes: int = 35) -> None:
    rng = random.Random(20261124)
    ahora = datetime.now(timezone.utc)
    inicio_ventana = ahora - timedelta(days=30)
    clientes = generar_clientes(rng, num_clientes)

    with psycopg.connect(DB_URL, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT codigo, nombre, precio, stock_actual, stock_reservado, imagen
                FROM public.productos
                WHERE activo = true AND precio > 0 AND stock_actual > 0
                ORDER BY codigo
                """
            )
            productos_rows = cur.fetchall()
            if not productos_rows:
                raise RuntimeError("No hay productos activos con stock en public.productos.")

            productos = [
                {
                    "codigo": r[0],
                    "nombre": r[1],
                    "precio": float(r[2]),
                    "stock_actual": int(r[3]),
                    "stock_reservado": int(r[4]),
                    "imagen": r[5] or "",
                }
                for r in productos_rows
            ]

            cur.execute("SELECT id FROM public.perfiles WHERE activo = true LIMIT 5")
            actores = [str(r[0]) for r in cur.fetchall()]
            if not actores:
                actores = ["CLIENTE WEB"]

            for c in clientes:
                cur.execute(
                    """
                    INSERT INTO public.clientes (id, documento, nombre, telefono, direccion, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (documento) DO UPDATE
                    SET nombre = EXCLUDED.nombre,
                        telefono = EXCLUDED.telefono,
                        direccion = EXCLUDED.direccion
                    RETURNING id
                    """,
                    (
                        c["id"],
                        c["documento"],
                        c["nombre"],
                        c["telefono"],
                        c["direccion"],
                        inicio_ventana + timedelta(seconds=rng.randint(1, 86400)),
                    ),
                )
                row_id = cur.fetchone()
                if row_id:
                    c["id"] = str(row_id[0])

            estados_posibles = ["DESPACHADO", "DESPACHADO", "CONFIRMADO", "BORRADOR", "CANCELADO"]

            for _ in range(num_pedidos):
                cliente = rng.choice(clientes)
                estado = rng.choice(estados_posibles)
                canal = "WEB" if rng.random() < 0.45 else "INTERNO"
                entrega = "DELIVERY" if rng.random() < 0.6 else "RECOJO"
                actor = "CLIENTE WEB" if canal == "WEB" else rng.choice(actores)
                actor_uuid = None if actor == "CLIENTE WEB" else actor

                offset_segundos = rng.randint(0, 30 * 24 * 3600)
                fecha_creacion = inicio_ventana + timedelta(seconds=offset_segundos)
                fecha_actualizacion = fecha_creacion + timedelta(minutes=rng.randint(5, 180))
                if fecha_actualizacion > ahora:
                    fecha_actualizacion = ahora

                pedido_id = f"PED-{fecha_creacion.strftime('%Y')}-{uuid.uuid4().hex.upper()}"
                op_id = str(uuid.uuid4())
                token_seg = str(uuid.uuid4())

                num_lineas = rng.randint(1, min(4, len(productos)))
                elegidos = rng.sample(productos, num_lineas)

                lineas_preparadas = []
                total_solicitado = 0.0
                total_efectivo = 0.0

                for prod in elegidos:
                    disponible = max(0, prod["stock_actual"] - prod["stock_reservado"])
                    if estado == "DESPACHADO":
                        if disponible < 1:
                            continue
                        qty = rng.randint(1, min(3, disponible))
                        qty_conf = qty
                        qty_prep = qty
                    elif estado == "CONFIRMADO":
                        if disponible < 1:
                            continue
                        qty = rng.randint(1, min(2, disponible))
                        qty_conf = qty
                        qty_prep = rng.randint(0, qty)
                    else:
                        qty = rng.randint(1, 4)
                        qty_conf = 0
                        qty_prep = 0

                    precio_u = round(prod["precio"], 2)
                    sub_sol = round(qty * precio_u, 2)
                    sub_ef = round(qty_conf * precio_u, 2)
                    total_solicitado = round(total_solicitado + sub_sol, 2)
                    total_efectivo = round(total_efectivo + sub_ef, 2)

                    lineas_preparadas.append(
                        (
                            prod,
                            qty,
                            qty_conf,
                            qty_prep,
                            precio_u,
                            sub_sol,
                            sub_ef,
                        )
                    )

                if not lineas_preparadas:
                    estado = "BORRADOR"
                    for prod in elegidos[:2]:
                        qty = rng.randint(1, 3)
                        precio_u = round(prod["precio"], 2)
                        sub_sol = round(qty * precio_u, 2)
                        total_solicitado = round(total_solicitado + sub_sol, 2)
                        lineas_preparadas.append((prod, qty, 0, 0, precio_u, sub_sol, 0.0))

                constancia = f"GUIA-{pedido_id}" if estado == "DESPACHADO" else None
                total_guardar = total_efectivo if estado in ("CONFIRMADO", "DESPACHADO") else total_solicitado

                cur.execute(
                    """
                    INSERT INTO public.pedidos (
                        id, cliente_nombre, cliente_documento, cliente_telefono, cliente_direccion,
                        estado, total, notas, actor, created_at, updated_at, canal, cliente_id,
                        version, total_solicitado, entrega, constancia, token_seguimiento, operacion_id
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        pedido_id,
                        cliente["nombre"],
                        cliente["documento"],
                        cliente["telefono"],
                        cliente["direccion"],
                        estado,
                        total_guardar,
                        "",
                        actor,
                        fecha_creacion,
                        fecha_actualizacion,
                        canal,
                        cliente["id"],
                        1,
                        total_solicitado,
                        entrega,
                        constancia,
                        token_seg,
                        op_id,
                    ),
                )

                for prod, qty, qty_conf, qty_prep, precio_u, _, sub_ef in lineas_preparadas:
                    sub_linea = sub_ef if estado in ("CONFIRMADO", "DESPACHADO") else round(qty * precio_u, 2)
                    cur.execute(
                        """
                        INSERT INTO public.pedido_lineas (
                            pedido_id, producto_id, producto_nombre, precio_unitario,
                            cantidad, subtotal, created_at, cantidad_confirmada,
                            cantidad_preparada, imagen
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            pedido_id,
                            prod["codigo"],
                            prod["nombre"],
                            precio_u,
                            qty,
                            sub_linea,
                            fecha_creacion,
                            qty_conf,
                            qty_prep,
                            prod["imagen"],
                        ),
                    )

                    if estado == "CONFIRMADO" and qty_conf > 0:
                        cur.execute(
                            """
                            INSERT INTO public.reservas (pedido_id, producto_codigo, cantidad)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (pedido_id, producto_codigo) DO UPDATE
                            SET cantidad = public.reservas.cantidad + EXCLUDED.cantidad
                            """,
                            (pedido_id, prod["codigo"], qty_conf),
                        )
                        prod["stock_reservado"] += qty_conf
                        cur.execute(
                            """
                            UPDATE public.productos
                            SET stock_reservado = %s,
                                version = version + 1,
                                updated_at = %s
                            WHERE codigo = %s
                            """,
                            (prod["stock_reservado"], fecha_actualizacion, prod["codigo"]),
                        )

                    elif estado == "DESPACHADO" and qty_conf > 0:
                        nuevo_saldo = prod["stock_actual"] - qty_conf
                        prod["stock_actual"] = nuevo_saldo
                        cur.execute(
                            """
                            UPDATE public.productos
                            SET stock_actual = %s,
                                stock = %s,
                                version = version + 1,
                                updated_at = %s
                            WHERE codigo = %s
                            """,
                            (nuevo_saldo, nuevo_saldo, fecha_actualizacion, prod["codigo"]),
                        )
                        cur.execute(
                            """
                            INSERT INTO public.movimientos (
                                producto_codigo, tipo, cantidad, saldo, actor,
                                referencia, proveedor, operacion_id, created_at
                            )
                            VALUES (%s, 'SALIDA', %s, %s, %s, %s, '', %s, %s)
                            ON CONFLICT (operacion_id, producto_codigo) DO NOTHING
                            """,
                            (
                                prod["codigo"],
                                qty_conf,
                                nuevo_saldo,
                                actor_uuid,
                                pedido_id,
                                op_id,
                                fecha_actualizacion,
                            ),
                        )

        conn.commit()


if __name__ == "__main__":
    poblar_base_datos(num_pedidos=220, num_clientes=35)