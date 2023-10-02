FROM python:3.8.18-bookworm

WORKDIR /root/code


RUN pip3 install dash
RUN pip3 install dash_bootstrap_components
RUN pip3 install scikit-learn==1.2.2
RUN pip3 install mlflow==2.7.1


COPY ./code /root/code/
CMD tail -f /dev/null