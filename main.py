from flask import Flask, request
from flask_restful import Resource, Api
import cv2
import numpy as np

app = Flask(__name__)
api = Api(app)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


class PeopleCounter(Resource):
    def get(self):
        img = cv2.imread("images/family-932245_1280.jpg")
        boxes, weights = hog.detectMultiScale(img, winStride=(5, 5))
        return {'count': len(boxes)}


class PeopleCounterWithLinkGiven(Resource):
    def get(self):
        url = request.args.get('url')
        if url:
            img2 = cv2.imread(url)
            boxes, weights = hog.detectMultiScale(img2, winStride=(5, 5))
            return {'count from link': len(boxes)}
        else:
            return {'error': 'url parameter not provided'}


class PeopleCounterWithImage(Resource):
    def post(self):
        if 'image' not in request.files:
            return {'error': 'No image provided'}
        img_file = request.files['image']
        if img_file.filename == '':
            return {'error': 'No image filename provided'}
        if img_file:
            img_data = img_file.read()
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            boxes, weights = hog.detectMultiScale(img, winStride=(5, 5))
            return {'count from image': len(boxes)}


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


api.add_resource(PeopleCounterWithLinkGiven, '/link')
api.add_resource(PeopleCounter, '/')
api.add_resource(PeopleCounterWithImage, '/image')
api.add_resource(HelloWorld, '/test')

if __name__ == '__main__':
    app.run(debug=True)
