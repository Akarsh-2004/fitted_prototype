import numpy as np
from typing import Dict, Any, Tuple
from pipeline.detectors.face_blur import face_blurrer
from pipeline.detectors.yolo_detector import yolo_detector

class SceneClassifier:
    def classify_scene(self, img: np.ndarray) -> Tuple[str, Dict[str, Any], np.ndarray]:
        """
        Classifies an uploaded image and applies face blurring if it's a worn outfit.
        Returns:
          - scene_type: 'flat_single', 'flat_multi', 'single_person', 'group_photo'
          - metadata: dict of counts (faces, people, garments)
          - processed_img: face-blurred image (if people are present), otherwise original
        """
        # 1. Detect and blur faces for privacy
        blurred_img, face_count = face_blurrer.blur_faces(img)
        
        # 2. Detect people
        people_dets = yolo_detector.detect_people(img)
        person_count = len(people_dets)
        
        # 3. Detect garments (only if no people are detected to save compute and avoid overlap confusions)
        garment_count = 0
        garment_dets = []
        if person_count == 0:
            garment_dets = yolo_detector.detect_flat_lay_garments(img)
            garment_count = len(garment_dets)
            
        # 4. Apply Scene Classification Routing Rules
        # Default classification
        scene_type = "flat_single"
        
        if person_count > 1:
            scene_type = "group_photo"
        elif person_count == 1:
            scene_type = "single_person"
        elif person_count == 0:
            if garment_count > 1:
                scene_type = "flat_multi"
            else:
                scene_type = "flat_single"
                
        metadata = {
            "face_count": face_count,
            "person_count": person_count,
            "garment_count": garment_count,
            "detected_people": people_dets,
            "detected_garments": garment_dets
        }
        
        # For person-worn scenarios, return the privacy face-blurred image
        final_img = blurred_img if person_count > 0 else img
        
        return scene_type, metadata, final_img

scene_classifier = SceneClassifier()
