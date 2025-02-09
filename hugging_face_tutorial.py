from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I love to hate you")