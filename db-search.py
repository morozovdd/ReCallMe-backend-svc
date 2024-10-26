from db.db_helper import FaceDatabase

# Initialize database
db = FaceDatabase()

# Add a patient
patient_id = db.add_patient("John", "Doe")

# Add a person with face embedding
person_id = db.add_person(
    patient_id=patient_id,
    first_name="Jane",
    last_name="Smith",
    relationship="daughter",
    face_embedding=your_face_embedding  # From your face recognition model
)

# Find similar faces
matches = db.find_similar_face(query_face_embedding)
if matches:
    print(f"Found match: {matches[0]['first_name']} {matches[0]['last_name']}")