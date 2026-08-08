def get_disease_info(disease):

    disease_data = {

        "Tomato___Late_blight":
        "Late blight is caused by fungal infection. Use fungicides and avoid excess moisture.",

        "Tomato___healthy":
        "The plant appears healthy.",

        "Potato___Early_blight":
        "Early blight causes dark spots on leaves. Remove infected leaves and use fungicide."
    }

    return disease_data.get(
        disease,
        "No detailed information available."
    )