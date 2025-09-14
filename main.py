import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
# from tensorflow.lite.python.interpreter import Interpreter
from fastapi.middleware.cors import CORSMiddleware
from fastapi_proxiedheadersmiddleware import ProxiedHeadersMiddleware

app = FastAPI(title="Buffalo Breed Recognition API!",
              description="Written for SIH-Project by Gradient-Gang Team",
              version="1.1.0"
             )

# Proxy headers handling
app.add_middleware(ProxiedHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"], # Allow all headers
)


GATEKEEPER_MODEL_PATH = "./GateKeeper_for_buffalo.tflite"
BREED_RECOGNIZER_MODEL_PATH = "./Breed_Recognizer_for_buffalo.tflite"


try:
    gatekeeper_interpreter = tf.lite.Interpreter(model_path=GATEKEEPER_MODEL_PATH)
    gatekeeper_interpreter.allocate_tensors()
    gatekeeper_input_details = gatekeeper_interpreter.get_input_details()
    gatekeeper_output_details = gatekeeper_interpreter.get_output_details()
    print("Gatekeeper model loaded successfully.")

    breed_interpreter = tf.lite.Interpreter(model_path=BREED_RECOGNIZER_MODEL_PATH)
    breed_interpreter.allocate_tensors()
    breed_input_details = breed_interpreter.get_input_details()
    breed_output_details = breed_interpreter.get_output_details()
    print("Breed Recognizer model loaded successfully.")

except Exception as e:
    print(f"Error loading models: {e}")
    
    raise HTTPException(status_code=500, detail="Failed to load TFLite models.")



def load_and_prep_image(image_bytes, img_shape=224, scale=False):
    """
    Reads in an image from bytes, turns it into a tensor and reshapes into
    specified shape (img_shape, img_shape, color_channels=3).
    """
    try:
        
        img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        
        img = tf.image.resize(img, size=[img_shape, img_shape])
        
        if scale:
            return img / 255.
        else:
            return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

class BreedPrediction(BaseModel):
    """Pydantic model for the response."""
    predicted_class: str
    confidence_score: float

def get_class_names():
    """Dependency to load class names from a JSON file."""
    try:
        with open("breeds_dict.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="breeds_dict.json file not found.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON in breeds_dict.json.")


@app.get("/")
async def root():
    return {"message": "Welcome to the Buffalo Breed Recognition API! Use the /recognize_breed endpoint to upload an image."}

@app.post("/recognize_breed", response_model=BreedPrediction)
async def recognize_breed(
    file: UploadFile = File(...),
    class_names: dict = Depends(get_class_names)
):
    """
    Analyzes an uploaded image to first detect if it's a buffalo,
    then predicts its specific breed.
    """
    
    image_bytes = await file.read()

    
    img_tensor = load_and_prep_image(image_bytes, scale=False)
    img_expanded = tf.expand_dims(img_tensor, axis=0)

    
    gatekeeper_interpreter.set_tensor(gatekeeper_input_details[0]['index'], img_expanded)
    gatekeeper_interpreter.invoke()

    
    gatekeeper_pred_prob = gatekeeper_interpreter.get_tensor(gatekeeper_output_details[0]['index'])

    
    gatekeeper_class_names = ['buffalo', 'not_buffalo']
    
    if gatekeeper_pred_prob[0][0] > 0.10: 
        raise HTTPException(status_code=400, detail="No Buffalo Detected. Please upload a correct image.")
    

    breed_interpreter.set_tensor(breed_input_details[0]['index'], img_expanded)
    breed_interpreter.invoke()

    
    pred_prob = breed_interpreter.get_tensor(breed_output_details[0]['index'])
    

    predicted_index = np.argmax(pred_prob)
    predicted_class = class_names[str(predicted_index)]
    confidence_score = float(np.max(pred_prob) * 100)
    
    return BreedPrediction(
        predicted_class=predicted_class,
        confidence_score=confidence_score
    )

# Load the JSON data
with open("breeds.json", "r") as f:
    breed_data = json.load(f)


@app.get("/buffalo_breeds/")
def get_buffalo_breeds():
    return breed_data

@app.get("/buffalo_breeds/{breed_name}")
def get_buffalo_breed(breed_name: str):
    # Normalize breed name for case-insensitive lookup
    normalized_breed_name = breed_name.replace(" ", "_").lower()
    
    # Find the breed in the data
    found_breed = None
    for breed, details in breed_data.items():
        if breed.replace(" ", "_").lower() == normalized_breed_name:
                found_breed = details
                break
    
    if found_breed:
        return found_breed
    else:
        raise HTTPException(status_code=404, detail=f"Buffalo breed '{breed_name}' not found.")




