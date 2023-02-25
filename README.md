-- Run the Program --

    First, import the packages:
        - tensorflow
        - tkinter
        - sklearn

    To run the program, simply just run the runner.py python script in this file. 
    It will take up to 15 seconds for the Neural Network to train, but afterwords
    there will be a questionnaire of symptoms displayed on a window. At the end there
    will be a message box of the probability of having each disease. 

-- Experimentation -- 
    
    I utilized a data set of not real world cases, but a result of a calculation
    by someone done online based on results on mayo clinic. Unfortunately, due to 
    versatility of diseases/symptoms, data collection method, and lower data for the Cold,
    I could not get more than a 93% accuracy for the network no matter what I tried. 
    However, overall the program works very well. To test this, I challenged this program 
    with my father who is a real life Physician. I took 5 patients from the data set and 
    explained the symptoms to my father as well as inputted the symptoms into the AI. 
    My father diagnosed 3/5 patients properly while the AI got 5/5. 

    I hope that I can recreate this project in the future with real world data
    and more diseases. 

-- Sources -- 

    UI heavy inspiration from freecodecamp.org
    https://www.freecodecamp.org/news/how-to-create-a-gui-quiz-application-using-tkinter-and-open-trivia-db/

    Dataset by Walter Conway from kaggle
    https://www.kaggle.com/datasets/walterconway/covid-flu-cold-symptoms

    