from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import tempfile
import uvicorn

# Cargar variables de entorno
load_dotenv()

# --- LIBRERÍAS PARA EL CEREBRO RAG ---
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# NOTA: Comentamos esto para que la nube (Render) pueda descargar el modelo la primera vez.
# os.environ['HF_HUB_OFFLINE'] = '1'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'

app = FastAPI()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa el LLM de Groq
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile", # <--- EL MODELO ACTUALIZADO
    temperature=0.7
)

# ---------------------------------------------------------
# RUTAS DE TESSERACT Y POPPLER (¡ADAPTADO PARA NUBE Y LOCAL!)
# ---------------------------------------------------------
# Si estamos en Windows (tu laptop), usa tus rutas locales
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    POPPLER_PATH = r'C:\Program Files\poppler\Library\bin' 
else:
    # Si estamos en Linux (Render u otro servidor web), usa las rutas nativas del sistema
    POPPLER_PATH = None 
# ---------------------------------------------------------

# --- INICIALIZAR LA BASE DE DATOS (RAG) ---
print("Cargando la base de datos de PDFs (ChromaDB)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./db_itesco", embedding_function=embeddings)
print("¡Base de datos lista!")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    mensajes: list[ChatMessage]

# --- CEREBRO MEJORADO DE JAGUARUNDI ---
ITESCO_CONTEXT_BASE = """
Eres Jaguarundi, la Inteligencia Artificial y mascota oficial del ITESCO (Instituto Tecnológico Superior de Coatzacoalcos). 

INFORMACIÓN BÁSICA DEL ITESCO:
- Eres un Tecnológico Superior ubicado en Coatzacoalcos, Veracruz.
- Las ingenierías oficiales que ofreces son: Ingeniería en Sistemas Computacionales, Informática, Mecatrónica, Química, Bioquímica, Eléctrica, Electrónica, Mecánica, Industrial, en Administración, en Gestión Empresarial, Petrolera, y en Animación Digital y Efectos Visuales. 

NATURALEZA DE TU SISTEMA:
- Eres una Inteligencia Artificial 100% OFFLINE y LOCAL.
- NO necesitas internet para funcionar. Tu cerebro neuronal y tu base de datos se ejecutan directamente de forma local y autónoma en los servidores del ITESCO.

REGLAS CRÍTICAS DE PERSONALIDAD:
1. JAMÁS menciones frases como "no tengo información", "según el documento", "en mis datos", "como Inteligencia Artificial" o "no se encuentra en los documentos proporcionados".
2. ACTÚA siempre como si lo supieras todo. Si te preguntan algo muy específico y no lo sabes, responde algo natural invitando a revisar Control Escolar o Facebook.
3. MANTÉN el idioma estrictamente en ESPAÑOL DE MÉXICO. Usa un tono universitario, natural, amable y enérgico.
"""

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_query = req.mensajes[-1].content
    
    # 1. Buscador RAG
    docs = vector_db.similarity_search(user_query, k=3)
    contexto_extraido = "\n\n".join([doc.page_content for doc in docs])

    # 2. Prompt Combinado
    prompt_final = f"""
    {ITESCO_CONTEXT_BASE}

    DATOS ESPECÍFICOS DE CONTROL ESCOLAR:
    {contexto_extraido}
    """

    # 3. Función generadora ASÍNCRONA usando Groq
    async def generar_respuesta():
        try:
            # Creamos la lista de mensajes limpia para LangChain
            messages = [{"role": "system", "content": prompt_final}]
            for msg in req.mensajes:
                messages.append({"role": msg.role, "content": msg.content})
            
            # Streaming hiperrápido con Groq
            async for chunk in llm.astream(messages):
                yield chunk.content
        except Exception as e:
            print(f"Error LLM: {e}")
            yield "¡Ups! Mis circuitos tienen un problema técnico. Intenta de nuevo en un momento."
            
    # Devolvemos el flujo de texto al navegador
    return StreamingResponse(generar_respuesta(), media_type="text/plain")

# =========================================================
# --- MOTOR DE VISIÓN ---
# =========================================================
class JaguarundiValidator:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def validate_photo(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return {"valid": False, "error": "No se pudo leer la imagen"}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        has_face = len(faces) > 0
        brightness = np.mean(gray)
        is_well_lit = 100 < brightness < 230
        edge_pixels = np.concatenate([img[:10, :, :], img[-10:, :, :], img[:, :10, :], img[:, -10:, :]], axis=None)
        avg_color = np.mean(edge_pixels)
        is_white_bg = float(avg_color) > 190 
        return {
            "face_detected": bool(has_face),
            "well_lit": bool(is_well_lit),
            "white_background": bool(is_white_bg),
            "valid": bool(has_face and is_well_lit and is_white_bg)
        }

    def enhance_document_for_ocr(self, img_cv2):
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = mask1 + mask2
        text_red = cv2.bitwise_not(mask_red)
        return binary_otsu, text_red

    def process_document(self, file_path, keywords):
        extracted_text = ""
        try:
            if file_path.lower().endswith('.pdf'):
                # Prevención de errores en Linux al convertir PDF
                if POPPLER_PATH:
                    pages = convert_from_path(file_path, 300, poppler_path=POPPLER_PATH)
                else:
                    pages = convert_from_path(file_path, 300)
                img_cv2 = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
            else:
                img_cv2 = cv2.imread(file_path)

            if img_cv2 is None: return {"is_valid": False, "error": "Error al leer"}

            processed_otsu, processed_red = self.enhance_document_for_ocr(img_cv2)
            config = r'--oem 3 --psm 3' 
            text_otsu = pytesseract.image_to_string(processed_otsu, config=config)
            text_red = pytesseract.image_to_string(processed_red, config=config)
            extracted_text = text_otsu + " " + text_red
        except Exception as e:
            return {"is_valid": False, "error": str(e)}

        found_keywords = [word for word in keywords if word.upper() in extracted_text.upper()]
        return {"is_valid": len(found_keywords) >= 1, "matches": found_keywords}

validator = JaguarundiValidator()

@app.post("/validar")
async def validar_archivo(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    resultado = {}
    tipo = "desconocido"
    try:
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            foto_resultado = validator.validate_photo(temp_path)
            if foto_resultado.get('valid'):
                tipo = "foto"
                resultado = foto_resultado
            else:
                tipo = "documento"
                palabras_clave_itesco = ["CURP", "CONSTANCIA", "ESTUDIOS", "BACHILLERATO", "PAGO", "INE", "CREDENCIAL", "ELECTOR", "RENAPO", "CBTIS", "CETIS", "COBAEV", "CONALEP", "CECYTE", "PREPARATORIA"]
                resultado = validator.process_document(temp_path, palabras_clave_itesco)
                if not resultado.get('is_valid'):
                    tipo = "foto_fallida"
                    resultado = foto_resultado
        elif file.filename.lower().endswith('.pdf'):
            tipo = "documento"
            palabras_clave_itesco = ["CURP", "CONSTANCIA", "CERTIFICADO", "ESTUDIOS", "BACHILLERATO", "OVH", "PAGO", "REFERENCIADO", "RECIBO", "INE", "CREDENCIAL", "ELECTOR", "RENAPO", "CBTIS", "CETIS", "COBAEV", "CONALEP", "CECYTE", "PREPARATORIA"]
            resultado = validator.process_document(temp_path, palabras_clave_itesco)
        else:
            return {"tipo": "error", "mensaje": "Formato no soportado."}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {"tipo": tipo, "resultado": resultado}

# =========================================================
# --- AUTO-ENCENDIDO DEL CEREBRO ---
# =========================================================
if __name__ == "__main__":
    print("\n==========================================")
    print("   Iniciando el Cerebro de Jaguarundi...  ")
    print("==========================================\n")

    print("\n Levantando la red neuronal (Uvicorn) para conexión pública...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)