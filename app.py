import streamlit as st
import os

# Langchain y Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# Importar st_tailwind
try:
    import st_tailwind as tw
except ImportError:
    tw = None 

# --- st.set_page_config() DEBE SER LO PRIMERO ---
st.set_page_config(
    page_title="Consultor de Emprendimientos IA", 
    layout="wide", 
    initial_sidebar_state="auto", 
    page_icon="🚀"
)

# --- Carga de API Key y Inicialización de LLM (fuera de main) ---
_llm_initialization_error = None 
GOOGLE_API_KEY = None
llm = None
model_name_to_use = "gemini-1.5-flash-latest"

# Determinar si estamos en Streamlit Cloud
# IS_STREAMLIT_CLOUD = os.getenv('STREAMLIT_SERVER_ öffentliche_PORT') is not None # Otra forma
IS_STREAMLIT_CLOUD = os.environ.get('STREAMLIT_SHARING_MODE') == 'true' or "SHARE_ streamlit_io" in os.environ.get("SERVER_SOFTWARE", "") or os.getenv("STREAMLIT_ENVIRONMENT") == "SHARE"


if IS_STREAMLIT_CLOUD and hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
    # print("Cargando API Key desde st.secrets (Streamlit Cloud)") # Para depuración
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    # print("No en Streamlit Cloud o GOOGLE_API_KEY no en st.secrets. Intentando .env local.") # Para depuración
    try:
        from dotenv import load_dotenv
        # Asegurarse de que el .env está en el mismo directorio que app.py o especificar la ruta
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
            temperature=0.4,
        )
    except Exception as e:
        _llm_initialization_error = e 
        llm = None
else:
    # Si GOOGLE_API_KEY sigue siendo None después de los intentos
    _llm_initialization_error = Exception("GOOGLE_API_KEY no fue encontrada ni en st.secrets ni en el entorno local (.env).") if not _llm_initialization_error else _llm_initialization_error


# --- Definiciones de Preguntas y Funciones Auxiliares ---
preguntas_emprendimiento = [
    {"id": 1, "texto": "Cuál es la idea de negocio?", "detalle": "(Definir claramente el producto o servicio que se ofrecerá y cómo se diferencia de la competencia?)"},
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

def obtener_feedback_gemini(texto_pregunta, respuesta_usuario, nombre_emprendimiento, nombre_emprendedor):
    global llm, _llm_initialization_error 
    if not llm:
        if _llm_initialization_error:
            return f"Error al inicializar el modelo de IA: {_llm_initialization_error}"
        elif not GOOGLE_API_KEY:
            return "Error: API Key de Google no configurada."
        else:
            return "Error: El modelo de lenguaje no está inicializado por una razón desconocida."

    if not respuesta_usuario or not respuesta_usuario.strip():
        return "Veo que no has ingresado una respuesta aún. Tómate tu tiempo para reflexionar sobre esta pregunta. ¿Qué ideas iniciales te vienen a la mente?"
    
    prompt_consultor = f"""
Eres un consultor de emprendimientos altamente experimentado, calmado y analítico.
Tu enfoque se basa firmemente en los principios de dos libros fundamentales: "Empieza con el porqué" de Simon Sinek y "Built to Last" de Jim Collins y Jerry Porras.

El emprendedor, {nombre_emprendedor if nombre_emprendedor else 'Emprendedor/a'}, está trabajando en su proyecto llamado "{nombre_emprendimiento if nombre_emprendimiento else 'su emprendimiento'}".
Actualmente está respondiendo a la pregunta del cuestionario: "{texto_pregunta}"
Su respuesta ha sido:
"{respuesta_usuario}"

Tu tarea es analizar su respuesta desde la perspectiva de los libros mencionados y guiarlo para que profundice en su pensamiento. NO le des respuestas hechas ni soluciones directas. En lugar de eso, formula preguntas claras y reflexivas que lo ayuden a desarrollar sus propias ideas.

Considera lo siguiente en tu análisis y preguntas de seguimiento:
- **Claridad del Porqué (Sinek):** ¿La respuesta refleja un entendimiento profundo del propósito, causa o creencia detrás de la idea o acción? Si no es así, ¿cómo puedes preguntarle para que explore su "Porqué"? Por ejemplo: "Interesante perspectiva, {nombre_emprendedor if nombre_emprendedor else 'Emprendedor/a'}. Yendo un paso más allá, ¿cuál es la creencia fundamental que impulsa esta idea de negocio?" o "Cuando piensas en el impacto a largo plazo de esto, ¿cuál es el 'porqué' más profundo que te motiva?".
- **Visión a Largo Plazo y Fundamentos (Collins & Porras):** ¿La respuesta considera elementos para construir una empresa duradera? ¿Hay indicios de valores fundamentales, un propósito que va más allá del dinero, o una visión audaz? Pregunta para explorar esto. Por ejemplo: "Pensando en los próximos 10-20 años, ¿cómo contribuye esta respuesta a la visión a largo plazo que tienes para {nombre_emprendimiento if nombre_emprendimiento else 'tu emprendimiento'}?" o "Si {nombre_emprendimiento if nombre_emprendimiento else 'tu emprendimiento'} tuviera que escribir sus valores fundamentales, ¿cómo se relacionaría esta respuesta con ellos?".
- **Profundidad y Especificidad:** Si la respuesta es vaga o superficial, pide ejemplos concretos o mayor elaboration. Por ejemplo: "Entiendo la idea general. ¿Podrías darme un ejemplo específico de cómo se manifestaría esto en la operación diaria de {nombre_emprendimiento if nombre_emprendimiento else 'tu emprendimiento'}?" o "Cuando mencionas 'mejorar la experiencia del cliente', ¿qué aspectos específicos de esa experiencia tienes en mente y cómo los abordarías de manera diferente?".
- **Coherencia:** ¿La respuesta es coherente con posibles respuestas anteriores o con los principios generales de un negocio con propósito y visión?

**Tu estilo de comunicación:**
- Calmado, analítico, reflexivo.
- Empático y alentador, pero firme en guiar hacia la profundidad.
- Usa un lenguaje claro y accesible.

**Formato de tu respuesta (Output):**
1. Un breve reconocimiento o comentario inicial sobre la respuesta del emprendedor (1-2 frases). Por ejemplo: "Gracias por compartir esto, {nombre_emprendedor if nombre_emprendedor else 'Emprendedor/a'}." o "Es un punto interesante el que mencionas sobre..."
2. Una o dos preguntas de seguimiento CLAVE (no más de dos). Estas preguntas deben estar diseñadas para estimular su pensamiento según los principios descritos.

**Importante:**
- Si la respuesta del emprendedor es particularmente clara, profunda y bien alineada con los principios, puedes felicitarlo brevemente antes de tu pregunta de seguimiento. Ejemplo: "Excelente reflexión, {nombre_emprendedor if nombre_emprendedor else 'Emprendedor/a'}. Es evidente que has pensado profundamente en [aspecto positivo]. Para llevarlo aún más lejos, ¿has considerado...?" o "Muy bien articulado. Se nota la conexión con [principio]. Ahora, ¿cómo podríamos explorar...?"
- Si la respuesta está muy incompleta o no es clara, tu pregunta principal debe ser una solicitud de clarificación o elaboración. No intentes analizar algo que no está ahí.

Ahora, analiza la respuesta proporcionada y genera tu feedback y preguntas de seguimiento para {nombre_emprendedor if nombre_emprendedor else 'el emprendedor'}.
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

    if not tw: 
        st.error("El componente st_tailwind no se pudo importar. Por favor, instálalo: pip install st-tailwind")
        st.stop()

    tw.initialize_tailwind()

    if 'nombre_emprendimiento' not in st.session_state: st.session_state.nombre_emprendimiento = ""
    if 'nombre_emprendedor' not in st.session_state: st.session_state.nombre_emprendedor = ""
    if 'info_inicial_guardada' not in st.session_state: st.session_state.info_inicial_guardada = False
    if 'pregunta_actual_idx' not in st.session_state: st.session_state.pregunta_actual_idx = 0
    if 'respuestas' not in st.session_state: st.session_state.respuestas = {}
    if 'feedback_consultor' not in st.session_state: st.session_state.feedback_consultor = {}
    if 'volver_a_resumen_despues_de_editar' not in st.session_state: st.session_state.volver_a_resumen_despues_de_editar = False
    if 'editando_pregunta_id' not in st.session_state: st.session_state.editando_pregunta_id = None

    # --- Manejo de errores de configuración de LLM y API Key ---
    # Comprobamos si la API Key se cargó correctamente y si el LLM se inicializó.
    if not GOOGLE_API_KEY:
        st.sidebar.error("API Key de Google no configurada.")
        # Solo muestra el error principal si el usuario aún no ha comenzado.
        if not st.session_state.info_inicial_guardada:
            mensaje_error_api_key = "CONFIGURACIÓN REQUERIDA: La API Key de Google no está configurada. La funcionalidad de IA estará desactivada. "
            if IS_STREAMLIT_CLOUD:
                mensaje_error_api_key += "Por favor, configúrala en los 'Secrets' de la aplicación en Streamlit Cloud."
            else:
                mensaje_error_api_key += "Por favor, asegúrate de que tu archivo .env local está correctamente configurado con GOOGLE_API_KEY."
            st.error(mensaje_error_api_key)
    elif not llm: # API Key podría estar, pero LLM no se inicializó
        if _llm_initialization_error:
            st.sidebar.error("Error al inicializar Gemini.")
            st.error(f"ERROR DE IA: No se pudo inicializar el modelo Gemini. Detalles: {_llm_initialization_error}")
        else: # Caso genérico si llm es None sin error específico (poco probable si API key está)
            st.sidebar.error("Modelo Gemini no disponible.")
            st.error("ERROR DE IA: El modelo de lenguaje no está disponible.")
    else: # Todo OK con la IA
        st.sidebar.success(f"Conectado a: {model_name_to_use}")
    
    # --- Flujo de la aplicación ---
    if not st.session_state.info_inicial_guardada:
        with tw.container(classes="p-6 md:p-8 max-w-xl mx-auto bg-white rounded-xl shadow-xl mt-10"):
            tw.write("👋 ¡Bienvenido/a Emprendedor/a!", classes="text-2xl font-bold text-primario-app mb-4")
            tw.write("Antes de comenzar, por favor indícanos un poco sobre ti y tu proyecto:", classes="text-texto-secundario mb-6")
            
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

            if tw.button("Comenzar Consultoría", key="comenzar_btn", classes="w-full bg-acento-app hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg focus:outline-none focus:shadow-outline text-lg"):
                if nombre_emp_input.strip() and nombre_emprendedor_input.strip():
                    st.session_state.nombre_emprendimiento = nombre_emp_input.strip()
                    st.session_state.nombre_emprendedor = nombre_emprendedor_input.strip()
                    st.session_state.info_inicial_guardada = True
                    st.session_state.pregunta_actual_idx = 0
                    st.session_state.respuestas = {}
                    st.session_state.feedback_consultor = {}
                    st.session_state.editando_pregunta_id = None 
                    st.session_state.volver_a_resumen_despues_de_editar = False 
                    st.rerun()
                else:
                    st.warning("Por favor, completa ambos campos para continuar.")
        return

    with tw.container(classes="max-w-4xl mx-auto p-4 md:p-6"):
        titulo_texto_visible = "🚀 Asistente para Emprendedores Exitosos"
        if st.session_state.nombre_emprendimiento:
            titulo_texto_visible = f"🚀 Asistente para {st.session_state.nombre_emprendimiento}"
        
        tw.write(titulo_texto_visible, classes="text-3xl lg:text-4xl font-bold text-primario-app my-6 text-center")

        saludo_html_texto_main = f"""
        Hola <strong class="font-semibold text-primario-app">{st.session_state.get('nombre_emprendedor', 'Emprendedor/a')}</strong>. Esta plataforma te guiará, con el apoyo de un consultor IA especializado,
        a través de preguntas clave para desarrollar y refinar los objetivos de tu empresa.
        Nos basaremos en los principios de <em class="italic">Empieza con el Porqué</em> (Simon Sinek) y <em class="italic">Built to Last</em> (Jim Collins & Jerry Porras).
        """
        st.markdown(f"<p class='text-lg text-texto-secundario mb-8 text-center'>{saludo_html_texto_main}</p>", unsafe_allow_html=True)

        if st.session_state.volver_a_resumen_despues_de_editar:
            st.session_state.pregunta_actual_idx = len(preguntas_emprendimiento)
            st.session_state.volver_a_resumen_despues_de_editar = False
            st.session_state.editando_pregunta_id = None 

        idx_pregunta_actual = st.session_state.pregunta_actual_idx

        if idx_pregunta_actual < len(preguntas_emprendimiento):
            pregunta_actual_obj = preguntas_emprendimiento[idx_pregunta_actual]
            
            with tw.container(classes="p-6 md:p-8 bg-white rounded-xl shadow-lg mb-10"):
                tw.write(f"Pregunta {pregunta_actual_obj['id']}/{len(preguntas_emprendimiento)}: {pregunta_actual_obj['texto']}",
                         classes="text-xl md:text-2xl font-semibold text-primario-app mb-3")
                tw.write(pregunta_actual_obj['detalle'], classes="text-sm text-texto-secundario mb-6 italic")

                respuesta_guardada = st.session_state.respuestas.get(pregunta_actual_obj['id'], "")
                respuesta_usuario_input = st.text_area(
                    "Tu respuesta:", value=str(respuesta_guardada), height=180,
                    key=f"respuesta_q{pregunta_actual_obj['id']}",
                    help="Escribe tu respuesta aquí y luego presiona 'Siguiente Pregunta'."
                )

                if pregunta_actual_obj['id'] in st.session_state.feedback_consultor:
                    feedback_msg = st.session_state.feedback_consultor[pregunta_actual_obj['id']]
                    container_classes = "p-4 mt-4 rounded-lg "
                    if feedback_msg.startswith("Se ha excedido la cuota"):
                        st.warning(feedback_msg)
                    elif feedback_msg.startswith(("Hubo un error", "Error:", "El servicio de IA no está disponible")):
                        st.error(feedback_msg)
                    elif feedback_msg == "Veo que no has ingresado una respuesta aún. Tómate tu tiempo para reflexionar sobre esta pregunta. ¿Qué ideas iniciales te vienen a la mente?":
                         with st.chat_message("ai", avatar="🧑‍🏫"):
                             tw.write(feedback_msg, classes=container_classes + "bg-yellow-100 text-yellow-800 border border-yellow-300")
                    elif feedback_msg != "No se proporcionó respuesta para analizar.":
                         with st.chat_message("ai", avatar="🧑‍🏫"):
                             tw.write(feedback_msg, classes=container_classes + "bg-feedback-info-bg text-feedback-info-text border border-blue-200")


                if tw.button("Siguiente Pregunta ❯", key=f"siguiente_q{pregunta_actual_obj['id']}",
                             classes="mt-6 w-full md:w-auto bg-acento-app hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg focus:outline-none focus:shadow-outline text-md"):
                    respuesta_actual_procesada = respuesta_usuario_input.strip()
                    st.session_state.respuestas[pregunta_actual_obj['id']] = respuesta_actual_procesada
                    
                    feedback_obtenido = ""
                    if llm and GOOGLE_API_KEY:
                        if respuesta_actual_procesada:
                            with st.spinner("El consultor IA está reflexionando sobre tu respuesta..."):
                                feedback_obtenido = obtener_feedback_gemini(
                                    pregunta_actual_obj['texto'],
                                    respuesta_actual_procesada,
                                    st.session_state.nombre_emprendimiento,
                                    st.session_state.nombre_emprendedor
                                )
                        else:
                            feedback_obtenido = "Veo que no has ingresado una respuesta aún. Tómate tu tiempo para reflexionar sobre esta pregunta. ¿Qué ideas iniciales te vienen a la mente?"
                    elif not respuesta_actual_procesada:
                         feedback_obtenido = "Veo que no has ingresado una respuesta aún. Tómate tu tiempo para reflexionar sobre esta pregunta. ¿Qué ideas iniciales te vienen a la mente?"
                    else:
                        feedback_obtenido = "El servicio de IA no está disponible para dar feedback en este momento."
                    
                    st.session_state.feedback_consultor[pregunta_actual_obj['id']] = feedback_obtenido

                    if st.session_state.editando_pregunta_id is not None:
                        st.session_state.volver_a_resumen_despues_de_editar = True
                    else:
                         st.session_state.pregunta_actual_idx += 1
                    st.rerun()
        else: 
            st.success(f"¡Excelente trabajo, {st.session_state.get('nombre_emprendedor', 'Emprendedor/a')}! Has completado todas las preguntas iniciales para {st.session_state.get('nombre_emprendimiento', 'tu emprendimiento')}.")
            st.balloons()
            
            tw.write(f"Resumen para {st.session_state.get('nombre_emprendimiento', 'tu Emprendimiento')}:",
                     classes="text-2xl lg:text-3xl font-bold text-primario-app mt-8 mb-6 text-center")

            if not st.session_state.respuestas:
                st.warning("Aún no has respondido ninguna pregunta.")
            else:
                for i, pregunta_info in enumerate(preguntas_emprendimiento):
                    with tw.container(classes="p-6 mb-6 bg-white rounded-xl shadow-lg"):
                        respuesta = st.session_state.respuestas.get(pregunta_info['id'], "*No respondida*")
                        feedback = st.session_state.feedback_consultor.get(pregunta_info['id'], "*Sin feedback aún.*")

                        tw.write(f"{pregunta_info['id']}. {pregunta_info['texto']}",
                                 classes=f"text-lg font-semibold text-primario-app mb-2")
                        st.markdown(f"<blockquote class='pl-4 italic border-l-4 border-gray-300 my-3 text-texto-secundario bg-gray-50 p-3 rounded-r-md'>{respuesta if respuesta.strip() else '*No respondida*'}</blockquote>", unsafe_allow_html=True)
                        
                        container_classes_resumen = "p-3 mt-3 rounded-lg "
                        if feedback != "*Sin feedback aún.*" and \
                           feedback != "No se proporcionó respuesta para analizar." and \
                           feedback != "El servicio de IA no está disponible para dar feedback en este momento.":
                             if feedback.startswith("Se ha excedido la cuota"):
                                st.warning(f"*Nota del sistema:* {feedback}")
                             elif feedback.startswith("Hubo un error") or feedback.startswith("Error:"):
                                st.error(f"*Nota del sistema:* {feedback}")
                             elif feedback == "Veo que no has ingresado una respuesta aún. Tómate tu tiempo para reflexionar sobre esta pregunta. ¿Qué ideas iniciales te vienen a la mente?":
                                with st.chat_message("ai", avatar="🧑‍🏫"):
                                    tw.write(feedback, classes=container_classes_resumen + "bg-yellow-100 text-yellow-800 border border-yellow-300")
                             else:
                                with st.chat_message("ai", avatar="🧑‍🏫"):
                                    tw.write(f"<span class='font-semibold'>Reflexión del Consultor IA:</span><br>{feedback}", classes=container_classes_resumen + "bg-feedback-info-bg text-feedback-info-text border border-blue-200")
                        
                        if tw.button(f"✏️ Editar Respuesta", key=f"edit_btn_{pregunta_info['id']}",
                                     classes="mt-4 text-sm bg-gray-100 hover:bg-gray-200 text-acento-app font-semibold py-2 px-4 border border-gray-300 rounded-md"):
                            st.session_state.pregunta_actual_idx = i
                            st.session_state.editando_pregunta_id = pregunta_info['id']
                            st.session_state.volver_a_resumen_despues_de_editar = False
                            st.rerun()
                    if i < len(preguntas_emprendimiento) -1 :
                        st.markdown("<div class='h-px bg-gray-200 my-6'></div>", unsafe_allow_html=True)
            
            with tw.container(classes="mt-10 text-center"):
                if tw.button("🏁 Completado y Conforme",
                            classes="w-auto bg-exito-app hover:bg-green-700 text-white font-bold py-3 px-8 rounded-lg text-lg"):
                    st.info("¡Genial! Has sentado una base sólida. En futuros módulos podremos profundizar aún más.")

                if tw.button("🔄 Reiniciar Todo el Cuestionario",
                            classes="mt-4 w-auto bg-peligro-app hover:bg-red-700 text-white font-bold py-3 px-8 rounded-lg text-lg"):
                    keys_to_delete = list(st.session_state.keys())
                    for key in keys_to_delete:
                        del st.session_state[key]
                    st.rerun()

if __name__ == "__main__":
    main()