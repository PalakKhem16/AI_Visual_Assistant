import openai
from typing import Dict, Any

class NLPProcessor:
    def __init__(self):
        pass
        
    def generate_answer(self, question: str, visual_context: str) -> str:
        """
        Generate an answer using GPT-4 based on the visual context and question.
        
        Args:
            question: The user's question about the scene
            visual_context: Description of what's detected in the current frame
        """
        try:
            # Create a prompt that combines the visual context and question
            prompt = f"""Visual Context: {visual_context}
            
Question: {question}

Based on the visual context provided, please answer the question."""

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that answers questions about what's visible in a scene. Be concise and focus on what's actually detected in the image."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error in generate_answer: {str(e)}")
            return "I'm sorry, I couldn't process your question at this time."

    def create_scene_summary(self, vision_data: Dict[str, Any]) -> str:
        """
        Create a natural language summary of the scene based on vision data.
        
        Args:
            vision_data: Dictionary containing detected objects and text
        """
        objects = vision_data.get('objects', [])
        text = vision_data.get('text', '')
        
        # Count objects by class
        object_counts = {}
        for obj in objects:
            class_name = obj['class']
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
        # Create summary
        summary_parts = []
        
        # Add object descriptions
        for class_name, count in object_counts.items():
            if count > 1:
                summary_parts.append(f"{count} {class_name}s")
            else:
                summary_parts.append(f"a {class_name}")
                
        # Combine object descriptions
        if summary_parts:
            summary = "I can see " + ", ".join(summary_parts[:-1])
            if len(summary_parts) > 1:
                summary += f" and {summary_parts[-1]}"
            elif len(summary_parts) == 1:
                summary = "I can see " + summary_parts[0]
        else:
            summary = "I don't see any recognizable objects"
            
        # Add text if found
        if text:
            summary += f". I also see text that reads: {text}"
            
        return summary
