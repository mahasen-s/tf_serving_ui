FROM tensorflow/serving:1.12.0-gpu

RUN mkdir -p /models/object-detect

EXPOSE 8080
ENTRYPOINT ["tensorflow_model_server", "--model_base_path=/models/object-detect"]
CMD ["--rest_api_port=8080", "--port=8081"]
