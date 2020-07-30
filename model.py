import kfserving
from typing import List, Dict
from PIL import Image
import base64
import io
import logging
import SimpleITK as sitk
import numpy as np
import requests, json
import sys, os
from enum import Enum
import cv2

class IsotropicStrategy(Enum):
    IsotropicToMin = 1
    IsotropicToMax = 2

def b64_filewriter(filename, content):
    string = content.encode('utf8')
    b64_decode = base64.decodebytes(string)
    fp = open(filename, "wb")
    fp.write(b64_decode)
    fp.close()

def resample_to_isotropic(image, strategy=IsotropicStrategy.IsotropicToMax):
    """ Resample to image to isotropic spacing

    :param image: Input image
    :type image: SimpleITK.Image
    :param strategy: Strategy to use the max spacing or min spacing
    :type strategy: IsotropicStrategy
    :return: Isotropic image
    :rtype: SimpleITK.Image
    """
    old_spacing = image.GetSpacing()
    if strategy == IsotropicStrategy.IsotropicToMax:
        new_spacing = tuple(np.full((len(old_spacing)), max(old_spacing)))
    else:
        new_spacing = tuple(np.full((len(old_spacing)), min(old_spacing)))

    factor = np.divide(old_spacing, new_spacing)
    old_size = image.GetSize()
    new_size = np.multiply(old_size, factor)
    slice_resampler = sitk.ResampleImageFilter()
    slice_resampler.SetSize(tuple(int(s) for s in new_size))
    slice_resampler.SetOutputSpacing(new_spacing)
    slice_resampler.SetOutputOrigin(image.GetOrigin())
    output_image = slice_resampler.Execute(image)
    return output_image

def segment_and_write(input_file, jpeg_output_file):
    output_file = jpeg_output_file.replace('.jpeg', '.mhd')
    print(output_file)
    model_file = './fluids_4_liverpool_cirrus.h5'
    os.system('momaku segmentation oct-segment-fluid {} {} {} 4'.format(input_file, output_file, model_file))

    print('Saving output {} as JPEG.'.format(output_file))

    itkimage = sitk.ReadImage(input_file)

    z = int(itkimage.GetDepth()/2)
    seg = sitk.LabelOverlay(itkimage[:,:,z], sitk.ReadImage(output_file)[:,:,z])
    seg = resample_to_isotropic(seg, strategy=IsotropicStrategy.IsotropicToMin)

    sitk.WriteImage(seg, jpeg_output_file)
    

model_name = os.getenv('MODEL_NAME',None)


class KFServingSampleModel(kfserving.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        self.ready = True

    def predict(self, inputs: Dict) -> Dict:
        del inputs['instances']
        logging.info("prep =======> %s",str(type(inputs)))
        try:
            json_data = inputs
        except ValueError:
            return json.dumps({ "error": "Recieved invalid json" })
        # Content sent by client
        data1 = json_data["signatures"]["inputs"][0][0]["data1"]
        data2 = json_data["signatures"]["inputs"][0][0]["data2"]
        
        # Writing files
        with open('images/original_sub_fourslice.mhd', 'w') as f:
            f.write(data1)

        b64_filewriter('images/original_sub_fourslice.raw',data2.split(',')[1])
        # with open('images/original_sub_fourslice.raw', 'wb') as f:
        #    f.write(data2.encode())
        # with open('images/original_sub_fourslice.raw', 'w') as f:
        #     f.write(data2)

        # Segmentation
        segment_and_write('images/original_sub_fourslice.mhd', 'images/original_sub_fourslice.jpeg')
        
       # Resize Image
        img = cv2.imread('images/original_sub_fourslice.jpeg')
        resized = cv2.resize(img, (768, 256), interpolation = cv2.INTER_AREA)
        cv2.imwrite('images/original_sub_fourslice.jpeg', resized)
       
       # Converting image to base64 string
        with open('images/original_sub_fourslice.jpeg', 'rb') as open_file:
            byte_content = open_file.read()
        base64_bytes = base64.b64encode(byte_content)
        base64_string = base64_bytes.decode('utf-8')

        return {"out_image":base64_string}

if __name__ == "__main__":
    model = KFServingSampleModel(model_name)
    model.load()
    kfserving.KFServer(workers=1).start([model])
