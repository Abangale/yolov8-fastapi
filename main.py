################################# IMPORT #################################
import json
import pandas as pd
from PIL import Image
import base64

from fastapi import FastAPI, File, status
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException

from app import get_image_from_bytes
from app import detect_sample_model
from app import add_bboxs_on_img
from app import get_bytes_from_image

############################# FastAPI Setup #############################

# title
app = FastAPI(
    title="Object Detection FastAPI Template",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="2023.1.31",
)

# This function is needed if you want to allow client requests 
# from specific domains (specified in the origins argument) 
# to access resources from the FastAPI server, 
# and the client and server are hosted on different domains.
origins = [
    "http://localhost",
    "http://localhost:8008",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def save_openapi_json():
    '''This function is used to save the OpenAPI documentation 
    data of the FastAPI application to a JSON file. 
    The purpose of saving the OpenAPI documentation data is to have 
    a permanent and offline record of the API specification, 
    which can be used for documentation purposes or 
    to generate client libraries. It is not necessarily needed, 
    but can be helpful in certain scenarios.'''
    openapi_data = app.openapi()
    # Change "openapi.json" to desired filename
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)

# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get('/healthcheck', status_code=status.HTTP_200_OK)
def perform_healthcheck():
    '''
    It basically sends a GET request to the route & hopes to get a "200"
    response code. Failing to return a 200 response code just enables
    the GitHub Actions to rollback to the last version the project was
    found in a "working condition". It acts as a last line of defense in
    case something goes south.
    Additionally, it also returns a JSON response in the form of:
    {
        'healtcheck': 'Everything OK!'
    }
    '''
    return {'healthcheck': 'Everything OK!'}


################################# Support Func #################################

def crop_image_by_predict(image: Image, predict: pd.DataFrame, crop_class_name: str,) -> Image:
    """Crop an image based on the detection of a certain object in the image.
    
    Args:
        image: Image to be cropped.
        predict (pd.DataFrame): Dataframe containing the prediction results of object detection model.
        crop_class_name (str, optional): The name of the object class to crop the image by. if not provided, function returns the first object found in the image.
    
    Returns:
        Image: Cropped image or None
    """
    crop_predicts = predict[(predict['name'] == crop_class_name)]

    if crop_predicts.empty:
        raise HTTPException(status_code=400, detail=f"{crop_class_name} not found in photo")

    # if there are several detections, choose the one with more confidence
    if len(crop_predicts) > 1:
        crop_predicts = crop_predicts.sort_values(by=['confidence'], ascending=False)

    crop_bbox = crop_predicts[['xmin', 'ymin', 'xmax','ymax']].iloc[0].values
    # crop
    img_crop = image.crop(crop_bbox)
    return(img_crop)


################################# MAIN Func #################################

@app.post("/img_object_detection")
def img_object_detection(mode: str, file: bytes = File(...)):
    '''
    Object Detection with output of Image BBox and JSON

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        dict: JSON Format data output.
    '''
    # Step 1: Get image from bytes
    input_image = get_image_from_bytes(file)

    # Step 2: Predict from model
    predict = detect_sample_model(input_image, mode)

    # Step 3: Add bbox on image
    final_image = add_bboxs_on_img(image = input_image, predict = predict)

    # Step 4: Count total fresh and no-fresh label
    total = len(predict['name'])
    fresh = len(predict[predict['name'] == 'fresh'])
    no_fresh = len(predict[predict['name'] == 'no-fresh'])

    # Step 5: Get bytes from image and format into base64
    image_bytes = get_bytes_from_image(final_image)
    bytes = image_bytes.getvalue()  # Read the contents of the BytesIO object as bytes
    image_result = {'image_data': base64.b64encode(bytes).decode('utf-8')}

    # Step 6: JSON Structure
    response_data = {
        'total':total,
        'fresh':fresh,
        'no_fresh':no_fresh,
        **image_result
    }

    # Return JSON
    return response_data