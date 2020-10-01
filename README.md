# Audio Sentiment Analysis
Preprocessing of audio files for sentiment analysis.

Models used for classification of audios:
* Neural Network
* SVM: sklearn.svm.SVC
* Random Forest Classifier

## Conclusions

The accuracy of the models is close to 100%. However, when a real audio file is tested the results are disappointing. A larger database with real and natural emotions should be used in order to be able to get accurate predictions under all circumstances.

## Author

* **David Andres**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* [Reconocimiento de Estados Emocionales de Personas Mediante la Voz Utilizando Algoritmos de Aprendizaje de Máquina](https://www.academia.edu/38038693/Reconocimiento_de_Estados_Emocionales_de_Personas_Mediante_la_Voz_Utilizando_Algoritmos_de_Aprendizaje_de_M%C3%A1quina)
* Database: [Toronto emotional speech set (TESS)](https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF)
* Confusion matrix for NN approach: [scikit-learn documentation example](https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html)