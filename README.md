## Train a XGBoost regression model on Amazon SageMaker, host inference on a Docker container running on Amazon ECS on AWS Fargate and optionally expose as an API with Amazon API Gateway

This repository contains a sample to train a regression model in Amazon SageMaker using SageMaker's built-in XGBoost algorithm on the California Housing dataset and host the inference on a Docker container running on Amazon ECS on AWS Fargate and optionally expose as an API with Amazon API Gateway.

### Overview

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) is a fully managed end-to-end Machine Learning (ML) service. With SageMaker, you have the option of using the built-in algorithms or you can bring your own algorithms and frameworks to train your models.  After training, you can deploy the models in [one of two ways](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html) for inference - persistent endpoint or batch transform.

With a persistent inference endpoint, you get a fully-managed real-time HTTPS endpoint hosted on either CPU or GPU based EC2 instances.  It supports features like auto scaling, data capture, model monitoring and also provides cost-effective GPU support using [Amazon Elastic Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html).  It also supports hosting multiple models using multi-model endpoints that provide A/B testing capability.  You can monitor the endpoint using [Amazon CloudWatch](https://aws.amazon.com/cloudwatch/).  In addition to all these, you can use [Amazon SageMaker Pipelines](https://aws.amazon.com/sagemaker/pipelines/) which provides a purpose-built, easy-to-use Continuous Integration and Continuous Delivery (CI/CD) service for Machine Learning.

There are use cases where you may want to host the ML model on a real-time inference endpoint that is cost-effective and do not require all the capabilities provided by the SageMaker persistent inference endpoint.  These may involve,
* simple models
* models whose sizes are lesser than 200 MB
* models that are invoked sparsely and do not need inference instances running all the time
* models that do not need to be re-trained and re-deployed frequently
* models that do not need GPUs for inference

In these cases, you can take the trained ML model and host it on a container on [Amazon ECS on AWS Fargate](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html) and optionally expose it as an API by front-ending it with a HTTP/REST API hosted on [Amazon API Gateway](https://aws.amazon.com/api-gateway/).  This will be cost-effective as compared to having real-time inference instances and still provide a fully-managed and scalable solution.

[Amazon Elastic Container Service (Amazon ECS)](https://aws.amazon.com/ecs) is a fully managed container orchestration service.  It is a highly scalable, fast container management service that makes it easy to run, stop and manage containers on a cluster. Your containers are defined in a task definition that you use to run individual tasks or tasks within a service.  In this context, a service is a configuration that enables you to run and maintain a specified number of tasks simultaneously in a cluster. You can run your tasks and services on a serverless infrastructure that is managed by [AWS Fargate](https://aws.amazon.com/fargate). Alternatively, for more control over your infrastructure, you can run your tasks and services on a cluster of Amazon EC2 instances that you manage.

This example demonstrates this solution by using SageMaker's [built-in XGBoost algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html) to train a regression model on the [California Housing dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html).  It loads the trained model as a Python3 [pickle](https://docs.python.org/3/library/pickle.html) object in a Python3 [Flask](https://flask.palletsprojects.com/en/1.1.x/) app script in a container to be hosted on [Amazon ECS on AWS Fargate](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html).  Finally, it provides instructions for exposing it as an API by front-ending it with a HTTP/REST API hosted on [Amazon API Gateway](https://aws.amazon.com/api-gateway/).

**Warning:** The Python3 [pickle](https://docs.python.org/3/library/pickle.html) module is not secure.  Only unpickle data you trust.  Keep this in mind if you decide to get the trained ML model file from somewhere instead of building your own model.

Note:

* This notebook should only be run from within a SageMaker notebook instance as it references SageMaker native APIs.
* If you already have a trained model that can be loaded as a Python3 [pickle](https://docs.python.org/3/library/pickle.html) object, then you can skip the training step in this notebook and directly upload the model file to S3 and update the code in this notebook's cells accordingly.
* In this notebook, the ML model generated in the training step has not been tuned as that is not the intent of this demo.
* In this notebook, we will create only one ECS Task.  In order to scale to more tasks, you have to create an [Amazon ECS Service](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs_services.html) made up of multiple tasks.  You can then setup [Load Balancing](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-load-balancing.html) and [Auto Scaling](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-auto-scaling.html).

### Repository structure

This repository contains

* [A Jupyter Notebook](https://github.com/aws-samples/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway/blob/main/notebooks/sm_xgboost_ca_housing_ecs_container_model_hosting.ipynb) to get started.

* [A Flask app inference script in Python3](https://github.com/aws-samples/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway/blob/main/notebooks/scripts/container_sm_xgboost_ca_housing_inference.py) that contains the code for performing inference.

* [The California Housing dataset in CSV format](https://github.com/aws-samples/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway/tree/main/notebooks/datasets).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

