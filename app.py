import streamlit as st
import os

# Langchain y Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# --- st.set_page_config() DEBE SER LO PRIMERO ---
st.set_page_config(
    page_title="Consultor de Emprendimientos IA",
    layout="wide",
    initial_sidebar_state="auto",
    page_icon="🚀"
)

# --- Función para cargar CSS local ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        if os.getenv("STREAMLIT_ENVIRONMENT") != "SHARE":
            st.error(f"Archivo CSS '{file_name}' no encontrado. Asegúrate de compilar Tailwind y que el archivo esté en la carpeta 'static'.")

# --- Carga de API Key y Inicialización de LLM ---
_llm_initialization_error = None
GOOGLE_API_KEY = None
llm = None
model_name_to_use = "gemini-1.5-flash-latest"

IS_STREAMLIT_CLOUD = os.environ.get('STREAMLIT_SHARING_MODE') == 'true' or \
                     "SHARE_ streamlit_io" in os.environ.get("SERVER_SOFTWARE", "") or \
                     os.getenv("STREAMLIT_ENVIRONMENT") == "SHARE"

if IS_STREAMLIT_CLOUD and hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    try:
        from dotenv import load_dotenv
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    except ImportError:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name_to_use,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.5, # Ajustado para permitir algo de creatividad en ejemplos
        )
    except Exception as e:
        _llm_initialization_error = e
        llm = None
else:
    _llm_initialization_error = Exception("GOOGLE_API_KEY no fue encontrada ni en st.secrets ni en el entorno local (.env).") if not _llm_initialization_error else _llm_initialization_error


# --- Definiciones de Preguntas ---
preguntas_emprendimiento = [
    {"id": 1, "texto": "Cuál es la idea de negocio?", "detalle": "(Define claramente el producto o servicio que se ofrecerá y cómo se diferencia de la competencia?)"},
    {"id": 2, "texto": "Quién o cuál es el público objetivo?", "detalle": "(identificar el perfil demográfico de tus clientes potenciales, sus necesidades y preferencias)"},
    {"id": 3, "texto": "Cuál es la propuesta de valor?", "detalle": "(Qué valor agregado se ofrecerá a los clientes? Por qué deberían elegir tu producto o servicio por encima de los otros?)"},
    {"id": 4, "texto": "Cuál es el modelo de negocio?", "detalle": "(Define cómo generarás ingresos ya sea a través de la venta de productos, servicios, publicidad, suscripciones, etc?)"},
    {"id": 5, "texto": "Cómo se piensa promocionar el negocio?", "detalle": "(Cuál es el plan de Marketing? cómo se llegará al público objetivo. Estrategias de marketing digital, redes sociales, publicidad local, relaciones públicas, etc )"},
    {"id": 6, "texto": "Quiénes serían los competidores?", "detalle": "(Identificar negocios similares, evaluar y analizar sus debilidades y fortalezas. Cómo podrías diferenciarte?)"},
    {"id": 7, "texto": "Hay regulaciones especiales para este negocio?", "detalle": "(Qué requisitos legales se necesitan? Permisos especiales. Cumplimientos tributarios.)"},
    {"id": 8, "texto": "Cómo se gestionará el negocio?", "detalle": "(Qué sistemas se implementarán? Sistemas o Departamentos de Marketing, Talento o Recurso Humano, Finanzas, Producto, Administración y operaciones, Inversiones)"},
    {"id": 9, "texto": "A dónde se quiere llegar?", "detalle": "(Visión, misión, metas, objetivos, propósito, planes a largo plazo)"},
    {"id": 10, "texto": "Cuál es la inversión o capital inicial?", "detalle": "(locales, equipos, inventario, tecnología, personal, etc)"}
]

def obtener_feedback_gemini(texto_pregunta, detalle_pregunta, respuesta_usuario, nombre_emprendimiento, nombre_emprendedor, edit_count):
    global llm, _llm_initialization_error
    if not llm:
        if _llm_initialization_error:
            return f"Error al inicializar el modelo de IA: {_llm_initialization_error}"
        elif not GOOGLE_API_KEY:
            return "Error: API Key de Google no configurada."
        else:
            return "Error: El modelo de lenguaje no está inicializado por una razón desconocida."

    if edit_count >= 2:
        return f"¡Gracias por tu esfuerzo y dedicación en esta pregunta, {nombre_emprendedor if nombre_emprendedor else 'Emprendedor/a'}! Has trabajado mucho en ella. Podemos continuar."

    prompt_consultor = f"""
Eres un consultor de emprendimientos muy amigable, paciente y extremadamente claro, como si estuvieras explicando conceptos de negocios a un amigo adolescente que está empezando. Tu objetivo principal es ayudarle a pensar con claridad y profundidad sobre cada aspecto de su idea.
El principio de "Empezar con el Porqué" de Simon Sinek (entender la razón fundamental, la causa o creencia detrás del negocio) es importante y debe estar de fondo, pero **tu prioridad es abordar la pregunta específica que se le hizo al emprendedor.**

El emprendedor se llama {nombre_emprendedor if nombre_emprendedor else 'tú'} y su proyecto es "{nombre_emprendimiento if nombre_emprendimiento else 'tu idea'}".
Está respondiendo a la pregunta: "{texto_pregunta}"
El detalle de la pregunta es: "{detalle_pregunta}"
Su respuesta ha sido: "{respuesta_usuario if respuesta_usuario.strip() else 'Parece que aún no has respondido o tu respuesta es muy breve.'}"

**Instrucciones para tu respuesta:**

A. **SI LA RESPUESTA DEL USUARIO ES NULA O MUY CORTA (ej. "no sé", "vender cosas", menos de 2-3 palabras con sentido):**
    1.  **Explica la Pregunta de Forma Sencilla:** Reformula la pregunta "{texto_pregunta}" en palabras muy simples. Explica qué tipo de información se busca con ella. Por ejemplo, si es sobre "Público Objetivo", explica qué significa eso.
    2.  **DA 2-3 EJEMPLOS CONCRETOS Y SENCILLOS** relevantes para la pregunta "{texto_pregunta}". Estos ejemplos deben ilustrar respuestas claras y bien pensadas a ESA PREGUNTA.
        *   Ejemplo para "¿Quién es tu público objetivo?": "Imagina que quieres vender patinetas. Un público podrían ser chicos y chicas de 13 a 18 años que aman el skate y buscan productos duraderos y con estilo. Otro podría ser adultos jóvenes que usan la patineta para moverse por la ciudad y buscan algo ligero y práctico. ¿Ves cómo son diferentes?"
    3.  **Pregunta Guía:** Termina con una pregunta amable que invite al usuario a pensar en su propia situación basándose en la explicación y los ejemplos. Ejemplo: "Pensando en tu idea de {nombre_emprendimiento if nombre_emprendimiento else 'tu proyecto'}, ¿quiénes crees que serían las personas más interesadas en lo que ofreces? ¿Cómo son?"

B. **SI LA RESPUESTA DEL USUARIO ES SUPERFICIAL O GENERAL (ej. tiene algunas palabras pero no profundiza, no es específica):**
    1.  **Reconocimiento Positivo:** Empieza con algo como: "¡Entendido! Mencionas que [resume brevemente su respuesta]. Es un buen punto de partida."
    2.  **Explicación de por qué se necesita más detalle PARA ESA PREGUNTA:** "Para que esta parte de tu plan sea realmente fuerte, ayuda mucho si somos un poco más específicos. Por ejemplo, si la pregunta es sobre 'Propuesta de Valor' y dices 'dar un buen servicio', eso es genial, pero muchas empresas intentan hacer eso."
    3.  **DA UN EJEMPLO CONCRETO de una respuesta más detallada o específica PARA LA PREGUNTA "{texto_pregunta}"**: "Una propuesta de valor más específica podría ser 'Ofrecemos el único servicio de reparación de bicicletas en el barrio que te devuelve la bici el mismo día y con una garantía de 30 días, porque entendemos que necesitas tu bici funcionando ya'. ¿Notas la diferencia en el detalle?"
    4.  **Pregunta Guía Específica:** Haz una pregunta que le ayude a añadir ese nivel de detalle o especificidad a SU respuesta actual. Ejemplo: "Volviendo a tu idea de [su respuesta], ¿qué detalles podrías añadir para que alguien entienda exactamente qué te hace diferente o especial en este punto?"
    5.  **(Opcional, si aplica y la pregunta lo permite) Conexión Sutil al "Porqué":** "A veces, pensar en tu 'Porqué' principal te puede ayudar a encontrar esos detalles. Si tu 'Porqué' es [ejemplo de porqué], ¿cómo se reflejaría eso en tu respuesta a '{texto_pregunta}'?" (Usa esto con moderación y solo si encaja naturalmente).

C. **SI LA RESPUESTA DEL USUARIO ES BUENA, DETALLADA O BIEN ENCAMINADA:**
    1.  **Felicitación Específica:** "¡Muy bien, {nombre_emprendedor if nombre_emprendedor else 'crack'}! Me gusta mucho cómo has explicado [menciona algo específico y positivo de su respuesta]. Se nota que le has dado vueltas."
    2.  **1 o 2 Preguntas de Profundización RELEVANTES A LA PREGUNTA ACTUAL:**
        *   Estas preguntas deben buscar más claridad, implicaciones o los siguientes pasos relacionados con lo que acaba de responder.
        *   **Solo si es natural y relevante para la pregunta actual**, una de estas preguntas podría explorar cómo su respuesta se alinea con su "Porqué" general (la razón fundamental de su emprendimiento). Ejemplo: "Excelente. Y pensando en esa [su respuesta específica], ¿cómo crees que esto refuerza o comunica el 'Porqué' principal de {nombre_emprendimiento if nombre_emprendimiento else 'tu proyecto'}?"
        *   Otras preguntas de profundización podrían ser: "¿Qué desafíos prevés al implementar esto que mencionas?" o "¿Cómo medirías el éxito de esta parte de tu plan?"

**Estilo General Constante:**
*   Amigable, paciente, claro, como un mentor joven.
*   Positivo y alentador.
*   Evita jerga.
*   **Enfócate en la pregunta actual.** El "Porqué" es un trasfondo, no el tema de cada respuesta.

Ahora, analiza la respuesta del usuario, la pregunta que se le hizo, y sigue las instrucciones (A, B, o C) para generar tu feedback y pregunta(s).
    """
    try:
        messages = [HumanMessage(content=prompt_consultor)]
        ai_response = llm.invoke(messages)
        return ai_response.content
    except Exception as e:
        print(f"Error en llamada a Gemini API: {e}")
        if "quota" in str(e).lower():
            return "Se ha excedido la cuota de uso gratuito de la IA. Por favor, inténtalo más tarde."
        return f"Hubo un error al procesar tu respuesta con la IA. El equipo técnico ha sido notificado."

def main():
    global llm, _llm_initialization_error, GOOGLE_API_KEY, model_name_to_use, IS_STREAMLIT_CLOUD

    local_css("static/style.css")

    # --- INICIALIZACIÓN DE SESSION_STATE ---
    if 'nombre_emprendimiento' not in st.session_state: st.session_state.nombre_emprendimiento = ""
    if 'nombre_emprendedor' not in st.session_state: st.session_state.nombre_emprendedor = ""
    if 'info_inicial_guardada' not in st.session_state: st.session_state.info_inicial_guardada = False
    if 'pregunta_actual_idx' not in st.session_state: st.session_state.pregunta_actual_idx = 0
    if 'respuestas' not in st.session_state: st.session_state.respuestas = {}
    if 'feedback_consultor' not in st.session_state: st.session_state.feedback_consultor = {}
    if 'volver_a_resumen_despues_de_editar' not in st.session_state: st.session_state.volver_a_resumen_despues_de_editar = False
    if 'editando_pregunta_id' not in st.session_state: st.session_state.editando_pregunta_id = None
    if 'edit_counts' not in st.session_state: st.session_state.edit_counts = {}

    # --- Manejo de errores de configuración ---
    if not GOOGLE_API_KEY:
        st.sidebar.error("API Key de Google no configurada.")
        if not st.session_state.info_inicial_guardada:
            mensaje_error_api_key = "CONFIGURACIÓN REQUERIDA: La API Key de Google no está configurada. La funcionalidad de IA estará desactivada. "
            if IS_STREAMLIT_CLOUD:
                mensaje_error_api_key += "Por favor, configúrala en los 'Secrets' de la aplicación en Streamlit Cloud."
            else:
                mensaje_error_api_key += "Por favor, asegúrate de que tu archivo .env local está correctamente configurado con GOOGLE_API_KEY."
            st.error(mensaje_error_api_key)
    elif not llm:
        if _llm_initialization_error:
            st.sidebar.error("Error al inicializar Gemini.")
            st.error(f"ERROR DE IA: No se pudo inicializar el modelo Gemini. Detalles: {_llm_initialization_error}")
        else:
            st.sidebar.error("Modelo Gemini no disponible.")
            st.error("ERROR DE IA: El modelo de lenguaje no está disponible.")
    else:
        st.sidebar.success(f"Conectado a: {model_name_to_use}")

    # --- Flujo de la aplicación ---
    if not st.session_state.info_inicial_guardada:
        st.markdown("<div class='p-6 md:p-8 max-w-xl mx-auto bg-fondo-contenedor rounded-xl shadow-xl mt-10 border border-borde-contenedor'>", unsafe_allow_html=True)
        st.markdown("<h1 class='text-2xl font-bold text-primario-app mb-4'>👋 ¡Bienvenido/a Emprendedor/a!</h1>", unsafe_allow_html=True)
        st.markdown("<p class='text-texto-secundario mb-6'>Antes de comenzar, por favor indícanos un poco sobre ti y tu proyecto:</p>", unsafe_allow_html=True)

        nombre_emp_input = st.text_input(
            "Nombre de tu Emprendimiento:",
            value=st.session_state.get("nombre_emprendimiento_temp", ""),
            key="nombre_emp_input_key"
        )
        nombre_emprendedor_input = st.text_input(
            "Tu Nombre:",
            value=st.session_state.get("nombre_emprendedor_temp", ""),
            key="nombre_emprendedor_input_key"
        )

        if st.button("Comenzar Consultoría", key="comenzar_btn_manual", type="primary", use_container_width=True):
            if nombre_emp_input.strip() and nombre_emprendedor_input.strip():
                st.session_state.nombre_emprendimiento = nombre_emp_input.strip()
                st.session_state.nombre_emprendedor = nombre_emprendedor_input.strip()
                st.session_state.info_inicial_guardada = True
                st.session_state.pregunta_actual_idx = 0
                st.session_state.respuestas = {}
                st.session_state.feedback_consultor = {}
                st.session_state.edit_counts = {}
                st.session_state.editando_pregunta_id = None
                st.session_state.volver_a_resumen_despues_de_editar = False
                st.rerun()
            else:
                st.warning("Por favor, completa ambos campos para continuar.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("<div class='max-w-4xl mx-auto p-4 md:p-6'>", unsafe_allow_html=True)

    titulo_texto_visible = "🚀 Asistente para Emprendedores Exitosos"
    if st.session_state.nombre_emprendimiento:
        titulo_texto_visible = f"🚀 Asistente para {st.session_state.nombre_emprendimiento}"

    st.markdown(f"<h1 class='text-3xl lg:text-4xl font-bold text-primario-app my-6 text-center'>{titulo_texto_visible}</h1>", unsafe_allow_html=True)

    saludo_html_texto_main = f"""
    Hola <strong class="font-semibold text-primario-app">{st.session_state.get('nombre_emprendedor', 'Emprendedor/a')}</strong>. Esta plataforma te guiará, con el apoyo de un consultor IA especializado,
    a través de preguntas clave para desarrollar y refinar los objetivos de tu empresa.
    Nos basaremos en el principio de <em class="italic text-texto-secundario">"Empezar con el Porqué"</em> de Simon Sinek.
    """
    st.markdown(f"<p class='text-lg text-texto-secundario mb-8 text-center'>{saludo_html_texto_main}</p>", unsafe_allow_html=True)

    if st.session_state.volver_a_resumen_despues_de_editar:
        st.session_state.pregunta_actual_idx = len(preguntas_emprendimiento)
        st.session_state.volver_a_resumen_despues_de_editar = False
        st.session_state.editando_pregunta_id = None

    idx_pregunta_actual = st.session_state.pregunta_actual_idx

    if idx_pregunta_actual < len(preguntas_emprendimiento):
        pregunta_actual_obj = preguntas_emprendimiento[idx_pregunta_actual]
        q_id = pregunta_actual_obj['id']

        st.markdown("<div class='p-6 md:p-8 bg-fondo-contenedor rounded-xl shadow-lg mb-10 border border-borde-contenedor'>", unsafe_allow_html=True)

        st.markdown(f"<h2 class='text-xl md:text-2xl font-semibold text-primario-app mb-1'>Pregunta {q_id}/{len(preguntas_emprendimiento)}:</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='text-lg md:text-xl text-texto-principal mb-3'>{pregunta_actual_obj['texto']}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p class='text-sm text-texto-secundario mb-6 italic'>{pregunta_actual_obj['detalle']}</p>", unsafe_allow_html=True)

        respuesta_guardada = st.session_state.respuestas.get(q_id, "")
        respuesta_usuario_input = st.text_area(
            "Tu respuesta:", value=str(respuesta_guardada), height=180,
            key=f"respuesta_q{q_id}",
            help="Escribe tu respuesta aquí y luego presiona 'Siguiente Pregunta'."
        )

        if q_id in st.session_state.feedback_consultor:
            feedback_msg = st.session_state.feedback_consultor[q_id]
            if feedback_msg.startswith("Se ha excedido la cuota"):
                st.warning(feedback_msg)
            elif feedback_msg.startswith(("Hubo un error", "Error:", "El servicio de IA no está disponible")):
                st.error(feedback_msg)
            elif feedback_msg == "Veo que no has ingresado una respuesta aún. Tómate tu tiempo para reflexionar sobre esta pregunta. ¿Qué ideas iniciales te vienen a la mente?":
                 with st.chat_message("ai", avatar="🧑‍🏫"):
                     st.markdown(f"<div class='p-4 mt-4 rounded-lg bg-yellow-100 text-yellow-800 border border-yellow-300 shadow'>{feedback_msg}</div>", unsafe_allow_html=True)
            elif feedback_msg.startswith("¡Gracias por tu esfuerzo y dedicación"):
                with st.chat_message("ai", avatar="🎉"):
                    st.markdown(f"<div class='p-4 mt-4 rounded-lg bg-green-100 text-green-800 border border-green-300 shadow'>{feedback_msg}</div>", unsafe_allow_html=True)
            elif feedback_msg != "No se proporcionó respuesta para analizar.":
                 with st.chat_message("ai", avatar="🧑‍🏫"):
                     st.markdown(f"<div class='p-4 mt-4 rounded-lg bg-feedback-info-bg text-feedback-info-text border border-blue-200 shadow'>{feedback_msg}</div>", unsafe_allow_html=True)


        if st.button("Siguiente Pregunta ❯", key=f"siguiente_q_manual{q_id}", type="primary", use_container_width=True):
            respuesta_actual_procesada = respuesta_usuario_input.strip()
            st.session_state.respuestas[q_id] = respuesta_actual_procesada

            edit_count_for_this_q = st.session_state.edit_counts.get(q_id, 0)
            feedback_obtenido = ""

            if llm and GOOGLE_API_KEY:
                with st.spinner("El consultor IA está reflexionando sobre tu respuesta..."):
                    feedback_obtenido = obtener_feedback_gemini(
                        pregunta_actual_obj['texto'],
                        pregunta_actual_obj['detalle'], 
                        respuesta_actual_procesada,
                        st.session_state.nombre_emprendimiento,
                        st.session_state.nombre_emprendedor,
                        edit_count_for_this_q
                    )
            elif not respuesta_actual_procesada:
                 feedback_obtenido = "Veo que no has ingresado una respuesta aún. Tómate tu tiempo para reflexionar sobre esta pregunta. ¿Qué ideas iniciales te vienen a la mente?"
            else:
                feedback_obtenido = "El servicio de IA no está disponible para dar feedback en este momento."

            st.session_state.feedback_consultor[q_id] = feedback_obtenido

            if st.session_state.editando_pregunta_id is not None:
                st.session_state.volver_a_resumen_despues_de_editar = True
            else:
                 st.session_state.pregunta_actual_idx += 1
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.success(f"¡Excelente trabajo, {st.session_state.get('nombre_emprendedor', 'Emprendedor/a')}! Has completado todas las preguntas iniciales para {st.session_state.get('nombre_emprendimiento', 'tu emprendimiento')}.")
        st.balloons()

        st.markdown(f"<h2 class='text-2xl lg:text-3xl font-bold text-primario-app mt-8 mb-6 text-center'>Resumen para {st.session_state.get('nombre_emprendimiento', 'tu Emprendimiento')}:</h2>", unsafe_allow_html=True)

        if not st.session_state.respuestas:
            st.warning("Aún no has respondido ninguna pregunta.")
        else:
            for i, pregunta_info in enumerate(preguntas_emprendimiento):
                q_id_resumen = pregunta_info['id']
                st.markdown("<div class='p-6 mb-6 bg-fondo-contenedor rounded-xl shadow-lg border border-borde-contenedor'>", unsafe_allow_html=True)
                respuesta = st.session_state.respuestas.get(q_id_resumen, "*No respondida*")
                feedback = st.session_state.feedback_consultor.get(q_id_resumen, "*Sin feedback aún.*")

                st.markdown(f"<h3 class='text-lg font-semibold text-primario-app mb-1'>{q_id_resumen}. {pregunta_info['texto']}</h3>", unsafe_allow_html=True)
                st.markdown(f"<blockquote class='pl-4 italic border-l-4 border-borde-contenedor my-3 text-texto-secundario bg-gray-50 p-3 rounded-r-md'>{respuesta if respuesta.strip() else '*No respondida*'}</blockquote>", unsafe_allow_html=True)

                if feedback != "*Sin feedback aún.*" and \
                   feedback != "No se proporcionó respuesta para analizar." and \
                   feedback != "El servicio de IA no está disponible para dar feedback en este momento.":
                     if feedback.startswith("Se ha excedido la cuota"):
                        st.warning(f"*Nota del sistema:* {feedback}")
                     elif feedback.startswith("Hubo un error") or feedback.startswith("Error:"):
                        st.error(f"*Nota del sistema:* {feedback}")
                     elif feedback == "Veo que no has ingresado una respuesta aún. Tómate tu tiempo para reflexionar sobre esta pregunta. ¿Qué ideas iniciales te vienen a la mente?":
                        with st.chat_message("ai", avatar="🧑‍🏫"):
                            st.markdown(f"<div class='p-3 mt-2 rounded-lg bg-yellow-100 text-yellow-800 border border-yellow-300 shadow'>{feedback}</div>", unsafe_allow_html=True)
                     elif feedback.startswith("¡Gracias por tu esfuerzo y dedicación"):
                        with st.chat_message("ai", avatar="🎉"):
                            st.markdown(f"<div class='p-3 mt-2 rounded-lg bg-green-100 text-green-800 border border-green-300 shadow'>{feedback}</div>", unsafe_allow_html=True)
                     else:
                        with st.chat_message("ai", avatar="🧑‍🏫"):
                            st.markdown(f"<div class='p-3 mt-2 rounded-lg bg-feedback-info-bg text-feedback-info-text border border-blue-200 shadow'><strong class='font-medium'>Reflexión del Consultor IA:</strong><br>{feedback}</div>", unsafe_allow_html=True)

                cols_edit_button = st.columns([0.8, 0.2])
                with cols_edit_button[0]:
                    if st.button(f"✏️ Editar Respuesta", key=f"edit_btn_manual_{q_id_resumen}", use_container_width=True):
                        st.session_state.edit_counts[q_id_resumen] = st.session_state.edit_counts.get(q_id_resumen, 0) + 1
                        st.session_state.pregunta_actual_idx = i
                        st.session_state.editando_pregunta_id = q_id_resumen
                        st.session_state.volver_a_resumen_despues_de_editar = False
                        st.rerun()
                with cols_edit_button[1]:
                    edit_count_for_q = st.session_state.edit_counts.get(q_id_resumen, 0)
                    if edit_count_for_q > 0:
                        st.markdown(f"<p class='text-xs text-gray-500 text-right pt-2'>Editado: {edit_count_for_q} {'vez' if edit_count_for_q == 1 else 'veces'}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                if i < len(preguntas_emprendimiento) -1 :
                    st.markdown("<div class='h-px bg-borde-contenedor my-6'></div>", unsafe_allow_html=True)

            st.markdown("<div class='mt-10 text-center space-y-4 md:space-y-0 md:flex md:justify-center md:space-x-4'>", unsafe_allow_html=True)
            if st.button("🏁 Completado y Conforme", type="primary", key="completado_final_manual"):
                st.info("¡Genial! Has sentado una base sólida. En futuros módulos podremos profundizar aún más.")
            if st.button("🔄 Reiniciar Todo el Cuestionario", key="reiniciar_final_manual"):
                keys_to_delete = list(st.session_state.keys())
                for key in keys_to_delete: del st.session_state[key]
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()