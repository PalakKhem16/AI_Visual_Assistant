import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
import cv2
from PIL import Image
import io

from utils.vision import VisionProcessor
from utils.audio import AudioProcessor
from utils.nlp import NLPProcessor

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Visual Assistant")

# Initialize processors
vision_processor = VisionProcessor()
audio_processor = AudioProcessor()
nlp_processor = NLPProcessor()

# Mount static directory for serving audio files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/process-frame/")
async def process_frame(file: UploadFile = File(...)):
    """
    Process an uploaded image frame and return scene description with audio.
    """
    try:
        # Read and process the image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        # Process the frame
        vision_results = vision_processor.process_frame(image)
        
        # Create scene description
        scene_description = nlp_processor.create_scene_summary(vision_results)
        
        # Convert description to speech
        audio_path = audio_processor.text_to_speech(scene_description)
        
        if not audio_path:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
            
        return {
            "description": scene_description,
            "audio_url": f"/static/output.mp3",
            "detected_objects": vision_results['objects'],
            "detected_text": vision_results['text']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice-command/")
async def process_voice_command(file: UploadFile = File(...)):
    """
    Process voice command and return spoken answer based on current visual context.
    """
    try:
        # Save uploaded audio file
        temp_path = audio_processor.save_uploaded_file(file)
        if not temp_path:
            raise HTTPException(status_code=400, detail="Failed to save audio file")
            
        # Transcribe audio to text
        question = audio_processor.transcribe_audio(temp_path)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        if not question:
            raise HTTPException(status_code=400, detail="Failed to transcribe audio")
            
        # Get current visual context
        # Note: In a real implementation, you might want to store the latest frame
        # or get a new frame from a connected camera
        scene_description = "Based on the latest frame: "  # Add actual context here
        
        # Generate answer using GPT-4
        answer = nlp_processor.generate_answer(question, scene_description)
        
        # Convert answer to speech
        audio_path = audio_processor.text_to_speech(answer)
        
        if not audio_path:
            raise HTTPException(status_code=500, detail="Failed to generate audio response")
            
        return {
            "question": question,
            "answer": answer,
            "audio_url": f"/static/output.mp3"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting server in debug mode...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="debug",
        timeout_keep_alive=60,
        workers=1
    )
