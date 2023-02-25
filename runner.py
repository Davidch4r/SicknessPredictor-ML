import os
# To silence non essential errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import tensorflow as tf
from tkinter import Tk, Canvas, StringVar, Label, Radiobutton, Button, messagebox
from sklearn.model_selection import train_test_split


"""
The network class is the Neural Network itself
It is created at the start of the program
"""

class Network():
    """
    First the data is loaded from the symptoms_data.csv data set
    Second the model is created 
    (it is preset to have 20 input and 4 output with 2 hidden layers but has some parameters)
    Third the model is trained, evaluated, and finalized with a softmax
    All these functions are specifically private
    """
    def __init__(self, shape, activation, learning_rate, batch_size, epochs):
        self.__load_data("symptoms_data.csv")
        self.__create_model(shape, activation, learning_rate)
        self.__finalize_model(batch_size, epochs)

    # Load Data
    def __load_data(self, filename):
        try:
            with open(filename) as f:
                reader = csv.reader(f)
                next(reader)

                # Organizing the data into a dictionary 
                self.data = []
                for row in reader:
                    self.data.append({
                        "evidence": [int(cell) for cell in row[:-1]],
                        "label": 0 if row[-1] == "ALLERGY" else 1 if row[-1] == "COLD" else 2 if row[-1] == "COVID" else 3
                    })

            # Seperating the values into training and testing
            self.evidence = [row["evidence"] for row in self.data]
            self.labels = [row["label"] for row in self.data]
            self.x_training, self.x_testing,self. y_training, self.y_testing = train_test_split(
                self.evidence, self.labels, test_size=0.4
            )

            print("Loaded Data Successfully")
        except:
            raise Exception("Count not load data")
    
    # Create the Model
    def __create_model(self, shape, activation, learning_rate):
        # Preset to 20 neuron input, 2 hidden layers, and 4 neuron output
        # However, shape, activation, and learning rate can be adjusted
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(20,)),
            tf.keras.layers.Dense(shape[0], activation=activation),
            tf.keras.layers.Dense(shape[1], activation=activation),
            tf.keras.layers.Dense(4)
        ])

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        metrics = ["accuracy"]

        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # Finalize the Model by
    # 1. Training
    # 2. Evaluating
    # 3. Adding Softmax
    def __finalize_model(self, batch_size, epochs):
        self.model.fit(self.x_training, self.y_training, batch_size=batch_size, epochs=epochs, verbose=2)

        self.model.evaluate(self.x_testing, self.y_testing, batch_size=batch_size, verbose=2)

        self.model = tf.keras.models.Sequential([
            self.model,
            tf.keras.layers.Softmax()
        ])

    # Converts the questionnaire answers from a list to a tensor
    # Then reshapes the tensor to fit the model network
    # Returns a prediction based on the network
    def predict(self, questionnaire):
        tensor = tf.constant(questionnaire)
        tensor = tf.reshape(tensor, [1, 20])
        return self.model.predict(tensor, verbose=2)

"""
A Questionnaire class to keep track of all the questions
and what question the user is currently on
"""
class Questionnaire:
    def __init__(self):
        # Possible symptoms tested
        self.symptoms = [
            "coughs",
            "muscle aches",
            "tiredness",
            "sore throat",
            "runny nose",
            "stuffy nose",
            "fever",
            "nausea",
            "vomitting",
            "diarrhea",
            "shortness of breath",
            "difficulty breathing",
            "loss of taste",
            "loss of smell",
            "itchy nose",
            "itchy eyes",
            "itchy mouth",
            "itchy inner ear",
            "sneezing",
            "pink eye"
        ]
        # Formats each symptom to "Do you have {Symptom}?"
        self.questions = [f'Do you have {symptom}?' for symptom in self.symptoms]
        # To keep track of current question
        self.index = 0

    # Gets the question at the current index
    # then increments the index by 1
    def next_question(self):
        question = self.questions[self.index]
        self.index += 1
        return question
    
    # Checks to see if not completed all questions
    def has_more_questions(self):
        return self.index < 20

"""
QuestionnaireUI is the overall UI/GUI of the code
Used to gather the user input and display the results
Heavily inspired by online tutorials
"""
class QuestionnaireUI:
    """
    Creates the interface with the
    Title, Question, Yes/No, and Next Button
    """
    def __init__(self, questionnaire, model):
        self.choices = []
        self.model = model
        self.q = questionnaire

        # The window and specifications
        self.window = Tk()
        self.window.title("Questionnaire")
        self.window.geometry("850x530")
        self.window.resizable(False, False)

        # Display Title
        self.display_title()

        # Creating a canvas with Question
        self.canvas = Canvas(width=800, height=250)
        self.question_text = self.canvas.create_text(
            400, 
            125,
            text="Question here",
            width=680,
            fill="#375362",
            font=('Ariel', 15, 'italic')
        )
        self.canvas.grid(row=2, column=0, columnspan=2, pady=50)
        self.display_question()

        # Declare a StringVar to store user's answer
        self.user_answer = StringVar()

        # Display the Yes/No using radio buttons
        self.opts = self.radio_buttons()
        self.display_options()

        # Next Button
        self.next_button()

        # Mainloop
        self.window.mainloop()

    # Displays the title on the top
    def display_title(self):
        title = Label(self.window, text="Questionnaire",
                        width=50, bg="purple", fg="white", font=("ariel", 20, "bold"))
        title.place(x=0, y=2)

    # Displays the question
    def display_question(self):
        q_text = self.q.next_question()
        self.canvas.itemconfig(self.question_text, text=q_text)
    
    # Yes/No Buttons
    def radio_buttons(self):
        # initialize the list with an empty list of options
        yes_no = []

        # Yes Button
        radio_btn = Radiobutton(self.window, text="", variable=self.user_answer, value='', font=("ariel", 14))
        yes_no.append(radio_btn)
        radio_btn.place(x=200, y=220)

        # No Button
        radio_btn = Radiobutton(self.window, text="", variable=self.user_answer, value='', font=("ariel", 14))
        yes_no.append(radio_btn)
        radio_btn.place(x=200, y=260)
        
        # return the Yes/No buttons
        return yes_no
    
    def display_options(self):
        """To display four options"""

        # deselecting the options at start
        self.user_answer.set(None)

        # Yes/No options' texts and values
        self.opts[0]['text'] = 'Yes'
        self.opts[0]['value'] = 1
        self.opts[1]['text'] = 'No'
        self.opts[1]['value'] = 0

    def next_button(self):

        # Next button to move to the next question
        next_button = Button(self.window, text="Next", command=self.nextbtn_func,
                             width=10, bg="purple", fg="white", font=("ariel", 16, "bold"))
        next_button.place(x=350, y=460)
    
    # Functionality for the next button
    def nextbtn_func(self):
        # Adds the choice to the choice list
        self.choices.append(self.user_answer.get())
        if self.q.has_more_questions():
            # Moves to next question
            self.display_question()
            self.display_options()
        else:
            # if no more questions, then it displays the results
            self.display_result()
            # destroys the self.window after leaving message box
            self.window.destroy()

    def display_result(self):
        # Predictions based on result
        predictions = self.model.predict([float(i) for i in self.choices])[0]
        # Dictionary to connect the prediction to the disease
        pred_dict = {
            "Allergy": predictions[0],
            "Cold": predictions[1],
            "Covid": predictions[2],
            "Flu": predictions[3]
        }
        # Sort prediction from least to greatest
        sorted_pred = dict(sorted(pred_dict.items(), key=lambda x:x[1]))
        # Display a message box of greatest to least predictions
        # Utilizes softmax to create each prediction as a percentage chance
        results = [f'Chance for {key}: {round(value * 100, 2)}%' for key, value in sorted_pred.items()]

        messagebox.showinfo("Result", f"{results[3]}\n{results[2]}\n{results[1]}\n{results[0]}")


# To run the program
def run():
    model = Network([80, 20], "sigmoid", 0.001, 30, 10)
    questionnaire = Questionnaire()
    QuestionnaireUI(questionnaire, model)
        


if __name__ == "__main__":
    run()