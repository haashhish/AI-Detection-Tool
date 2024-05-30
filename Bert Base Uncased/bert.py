#Updates to keyboard shortcuts … On Thursday, August 1, 2024, Drive keyboard shortcuts will be updated to give you first-letters navigation.Learn more
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import csv
import plotly.graph_objects as go




# Set up parameters
bert_model_name = 'bert-base-uncased'
num_classes = 2
max_length = 256
batch_size = 16
num_epochs = 10
learning_rate = 2e-5


def load_imdb_data(data_file):
    df = pd.read_csv(data_file)
    texts = df['text'].tolist()
    labels = df['label'].tolist()  # Assuming 'label' column contains the labels directly
    return texts, labels



data_file = "finaldataset.csv"
texts, labels = load_imdb_data(data_file)



class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}
    



# class BERTClassifier(nn.Module):
#     def __init__(self, bert_model_name, num_classes):
#         super(BERTClassifier, self).__init__()
#         self.bert = BertModel.from_pretrained(bert_model_name)
#         self.dropout = nn.Dropout(0.3)
#         self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

#     def forward(self, input_ids, attention_mask):  # Correct indentation here
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.pooler_output
#         x = self.dropout(pooled_output)
#         logits = self.fc(x)
#         return logits
    
    
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        # Include attention dropout in the configuration
        config = BertConfig.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name, config=config)
        # self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        # x = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits





def train(model, data_loader, optimizer, scheduler, device):
    model.train()  # Set the model to training mode
    total_loss = 0
    correct_predictions = 0
    total_examples = 0

    for batch_idx, batch in enumerate(data_loader):
        optimizer.zero_grad()  # Clear gradients before each optimization step
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Get model outputs, which are the logits in this case
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # Calculate loss using the logits and actual labels
        loss = nn.CrossEntropyLoss()(logits, labels)



        # l2_reg_loss = sum(torch.norm(param) ** 2 for param in model.parameters())
        # loss += 0.5 * 0.05 * l2_reg_loss
        total_loss += loss.item()

        # Calculate accuracy
        _, predicted_labels = torch.max(logits, dim=1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_examples += labels.size(0)
        loss.backward()  # Backpropagate the error
        optimizer.step()  # Update parameters
        scheduler.step()  # Update learning rate

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(data_loader)}: Loss {loss.item()}")

    average_loss = total_loss / len(data_loader)
    train_accuracy = correct_predictions / total_examples

    return average_loss, train_accuracy


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            # l2_reg_loss = sum(torch.norm(param) ** 2 for param in model.parameters())
            # loss += 0.5 * 0.05 * l2_reg_loss
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
            
    accuracy = accuracy_score(actual_labels, predictions)
    average_loss = total_loss / len(data_loader)
    return accuracy, average_loss,classification_report(actual_labels, predictions)


def predict_text_source(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    return "AI-generated" if preds.item() == 1 else "Human-written"



train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.3, random_state=42)

tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on : ",device)
model = BERTClassifier(bert_model_name, num_classes).to(device)
#model.load_state_dict(torch.load("bert_classifier.pth"))


optimizer = AdamW(model.parameters(), lr=learning_rate,weight_decay=0.05)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_metric = float('-inf')  # Initialize best validation metric (can be accuracy or loss)
patience = 3  # Number of epochs to wait for improvement

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss,train_accuracy = train(model, train_dataloader, optimizer, scheduler, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_accuracy, val_loss,report = evaluate(model, val_dataloader, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    print(report)
    # if val_accuracy > best_val_metric:  # Replace 'val_accuracy' with loss if needed
    #     best_val_metric = val_accuracy
    #     patience_counter = 0  # Reset patience counter
    # else:
    #     patience_counter += 1

    # if patience_counter >= patience:
    #     print(f"Early stopping triggered after {epoch + 1} epochs")
    #     break


# torch.save(model.state_dict(), "bert_classifier.pth")




# essay="""
# "Time spent with cats is never wasted." (Freud). Cats are the second most spread pets globally, with a population of 373 million pet cats in the U.S alone in 2018, according to Statista. Still, the question stands of how these cute pets that share about 95.6 percent of their DNA with tigers (Sung), one of the most aggressive wild animals today, became widespread in the world from ages came to live with us under one roof and be our best companion in our homes. The answer to this question lies in the word "domestication." Before answering this question, the meaning of domestication should be known. First, domestication is to tame an animal, especially by generations of breeding, to live in close association with human beings as a pet or work animal and usually creating a dependency so that the animal loses its ability to live in the wild (Harvard Dictionary). Now that domestication is defined by its broad meaning, and it can be limited to the process of cats' domestication of cats happened in two phases in two near places, the near eastern and ancient Egypt; nonetheless, there was a common misconception about the original location of cats' domestication among the scientific community due to a missing link in the fossil record of the cat's domestication history, leading to falsely thinking that ancient Egypt was the origin of cat domestication. This paper will argue against the old theory that cats were initially tamed in ancient Egypt and will serve to prove the opposite of the theory.
         
#  The first reason why domestication was not domesticated initially in Egypt is that the ascendant of all the modern domestic cats, Felis Sylvestris, "The Near Eastern wildcat Felis silvestris lybica is the only subspecies of wildcat that has been domesticated. It is native to Northern Africa and the Near East. This subspecies is the ancestor of all modern domestic cats, Felis silvestris catus" ( Krajcarz,17710-17711) However, it did not originally live in Egypt; as a matter of fact, it did not live in Africa at all (Kitchener,320-321), but a subspecies from that cat, called F.S Lybica, that started to roam in Egypt and in small populations that the modern cats derived from and were domesticated by humans. This early domestication happened with the first humans in the Near East, especially in southeast Turkey, who discovered agriculture and the urge to settle in one place to look after crops. Still, with seasons changing, the yields varied and affected the food chain of the early human, so humans needed a place to store their grains to subsist on them later in other seasons; however, with these places to keep food, mice and other animals that fed on the grains began to invade those storages causing damage to the stored food. Noticing that cats feed on these small animals, the early humans began to develop a relationship with them." Scientists, therefore, assume that African wildcats were attracted to first human settlements in the Fertile Crescent by rodents, in particular the house mouse" (Wilson,532-533). "The available archaeological evidence indicates that the process of wildcat domestication began in the Neolithic in the same place and time as the development of year-round settlements and the onset of an agricultural economy" (Driscoll, 37-39). This early relationship between early humans and cats developed a long time ago, as Driscoll said: "Considering the broadest range of dates for domestication to be from 11,000 to 4,000 B.P" (Driscoll, 37-39). This serves to prove that in geographic manners, cats were not domesticated initially in ancient Egypt but in areas close to it in different periods. 
# Some people believe that Egypt was the original place of cats taming. But what made people think that ancient Egypt is associated with the taming of cats? The answer to that question is quite simple. It lies in the religion and burying rituals of the old Egyptian culture as the cats were considered from the Egyptian religion pantheon; there was a goddess called Bastet worshipped in the form of cats(Engels). Also, ancient Egyptians used to mummify cats and bury them with the pharaoh. The first evidence of their theory is that historians said that the oldest cat painting was found in ancient Egypt about 3,600 years ago. Therefore, the ancient Egyptians must be the first humans to tame wild cats and make them pets. However, this was a common misconception among not only historians only, but even the scientific community. Because of the cat anatomy, where the bones are designed to be light to minimize the air friction and give the cats their speed (Travis,130-134), the bones tend typically to decompose faster, which made it harder to determine whether the skeletons of the cats belong to a domesticated or a wild cat as mentioned before cats kept a lot of properties from their wild ancestor. Hence, there is a missing link in the cats' fossil record, which made a false claim that cats were initially domesticated in ancient Egypt. "Until recently, the cat was commonly believed to have been domesticated in ancient Egypt" (Driscol,39-42). Furthermore, the actual oldest evidence of cat domestication was found around 9,500 years ago in Cyprus island as the archaeologists found the ancient burial in 2004. The archaeologists found a skeleton of an adult human and eight months old cat buried next to each other (Rothwell,1714,1715). 
# The second refute to the claim of cats being domesticated in ancient Egypt is that simply ancient Egypt is younger than the oldest fossil found. As the civilization in ancient Egypt started by king Narmer unifying the states at 3100 B.C and ended in the reign of pharaoh Pepi II around 2150 B.C with the invasion of Alexander the Great of Egypt. A civilization that lasted about 30 centuries (Van,57-58). 
# This means that ancient Egypt cannot be the initial place of the cat domestication process as it was not formed yet by the time the actual domestication was happening in the Near Eastern. The previously mentioned refutes serve to prove that historically, the ancient Egyptian civilization was not the first civilization to domesticate cats, but it was falsely thought to be because of a missing link in the fossil record that caused a common misconception between the historians and the scientific community. 

# The third reason why cats were not domesticated in Egypt is that the original domestication place of cats is still unknown until now. If the oldest fossil of a domesticated cat in Cyprus was around 10,000 years ago on the island, who brought the cats to this island? Cyprus is an island in the middle of the Mediterranean Sea. The nearest land to it is the Turkish coast, which is about 455 km away from the island. Furthermore, the longest distance that cats can swim without drowning is about 55 km (Gabbatiss). Accordingly, there was no way for the wild cat to cross that vast distance of the water independently, indicating that these cats needed the help of human companionship to get into ships with the humans. "Cats and small domestic dogs were brought from the mainland." (Vigne,8445-8449), so if cats were brought to the land by humans, who were the ones that brought the cats to this island, and why did they bring them to that island in particular?  The answer to these questions is that they were different generations of people "Recent discoveries indicated that Cyprus was frequented by Late PPNA people, but the earliest evidence until now for both the use of cereals and Neolithic villages on the island dates to 10,400 years ago" (Vigne,8445-8449)  these migrations or trips were to escape the shrinking coastal plains and the lousy weather (Fisher,3-15) then as generations passed, the early settlers eventually chose to live in that island because it was easier to find food and shelter, but why were cats brought with the early human travelers to this island? Because, as previously mentioned, cats hunt mice that can threaten the food storage of humans. Actual evidence of grains was found in the fossil record of the discovered burial in Cyprus, indicating that cats actually fed on smaller animals that subsisted or scavenged on the human offal or ate from the crops of humans (Pickrell). This example serves to prove that cats' initial domestication credit cannot be given entirely to ancient Egypt because, as a matter of fact, the people who brought the first domesticated cat to the island cannot be determined precisely from the three kinds of people who visited the island and only one of them decided to live on the island (Simmons,88-89). So, from all of these uncertainties and ambiguity, it cannot be determined whether one of the three types of settlers who came to the island was the first human to begin the domestication process of the cats as this kind may have gotten it from another area or different humans as the case with Cyprus and ancient Egypt because at every time period a place was thought to be the original place of domestication until a new fossil record is discovered that clears the opacity around the mystery of cats domestication. 
# The fourth reason why cats were not initially domesticated in ancient times in Egypt was that Egypt actually was the last stage of domestication that cats have gone through to reach their current ancestor of domestic cats. It can be noticed that modern domestic cats kept a lot of genetic and behavioral traits from their wild ancestors than most other domesticated animals do, meaning that there was an interbreed between domestic cats and surrounding wild cat populations (Montague, 17230-17235). It is thought that feral cats actually got domesticated twice in two different time periods over the course of the cats' domestication history. The first domestication is believed to happen in southwest Asia about 10,000 years ago, and the other one in Egypt about 3,500 years ago (Hunter,201-202). This is based on the analysis of modern cats' genome and DNA, which suggest that two different species of cats with two diverse source populations gave rise to the current gene pool of cats at two other times. That means that there are multiple points of domestication in cats' history. For example, the excavation evidence that supports this theory is that in Egypt, six burials have been found uncovered in the site of Heirkonpolis, dating back to between 3,600 and 3,800 years ago. Containing the bones of four kittens, their bones matched those of modern domestic and two adult cats, and one of the adult cats' bones was fractured and healing showing that it was cared for by its human companions (Van Neer). The relationship between cats and humans was so crucial that ancient Egyptians painted it on the walls, and the iconography showed cats alongside people or cats often hunting and eating smaller animals. But, if ancient Egypt was the last place to subdue cats, how did cats manage to spread to most of the world? Cats were brought to Rome by early Greek settlers and interaction between Rome and Egypt (Ottoni,1-7), with cultivations expanding and conquering each other around 2,000 years ago, especially within the Roman empire, cats followed along with their humans and eventually spreading worldwide reaching a population up to 600 million cats in the world, "The domestic cat is a popular pet species, with as many as 600 million individuals worldwide" (Montague, 17230-17235) because. These claims and facts highlight the fact that archaeologically, geographically, and historically, ancient Egypt was not the origin of cats' domestication because ancient Egypt was the second stop in cat's domestication history as it gave the cats its modern shape, identity and introduced them to all the world through its trade routes and interactions with other civilizations as Rome and Greek that made it possible for the cats to get that worldwide spread. Perhaps the great rule of Egypt in the domestication of cats' process as it showed great interest in cats was the cause of people getting the wrong ideas that Egypt was the initial place of cats taming in the first place. 
# In conclusion, it cannot be denied that ancient Egypt had a great rule in the domestication of cats throughout the course of human and cat history both alike; however, because of a missing link in the fossil record of cats, the anatomy of the cat and the structure of the cats' bones which decomposes fast making it harder for archaeologists to differentiate between the domestic cat and wild cat and finding evidence of the first place where cat domestication happened, there was common claim among the scientific community and historians that cats were initially tamed in ancient Egypt, which was proven by various ways that this claim is just a misconception because of the ambiguous and unclear history of cat domestication. Furthermore, it was proven historically that the initial domestication did not occur in Egypt as Egypt was not formed yet when was early signs of cat taming were happening on Cyprus island. Geographically, by showing that the ancestors of the cat, which is domesticated in Egypt, did not even live in Africa. Archeologically as it was shown that the oldest fossil of cats found in the burials dated back between 3,600 to 3,800 years ago, which relatively short period of time when compared to the fossils found on Cyprus, finally, the fact that it is not known yet who even brought cats to Cyprus in the first place as different generations traveled to the island, and any generation could have brought them, considering all of these factors, it can be concluded that ancient Egyptians were not the first humans to domesticate cats.
# """

# result=predict_text_source(essay, model, tokenizer, device)
# print(result)

epochs = range(1, num_epochs + 1)
# Create traces for training and validation loss
trace1 = go.Scatter(
    x=list(epochs),
    y=train_losses,
    mode='lines+markers',
    name='Training Loss',
    marker=dict(color='blue')
)

trace2 = go.Scatter(
    x=list(epochs),
    y=val_losses,
    mode='lines+markers',
    name='Validation Loss',
    marker=dict(color='red')
)

# Create traces for training and validation accuracy
trace3 = go.Scatter(
    x=list(epochs),
    y=train_accuracies,
    mode='lines+markers',
    name='Training Accuracy',
    marker=dict(color='purple')
)

trace4 = go.Scatter(
    x=list(epochs),
    y=val_accuracies,
    mode='lines+markers',
    name='Validation Accuracy',
    marker=dict(color='green')
)


# Create the figure and add traces for loss
fig = go.Figure()
fig.add_trace(trace1)
fig.add_trace(trace2)

# Set layout for loss plot
fig.update_layout(
    title='Training and Validation Loss',
    xaxis_title='Epoch',
    yaxis_title='Loss',
    legend_title='Legend',
    width=600,  # adjust size as needed
    height=400
)

# Show the figure
fig.show()

# Create a new figure for accuracy
fig2_accuracy = go.Figure()
fig2_accuracy.add_trace(trace3)
fig2_accuracy.add_trace(trace4)

# Set layout for accuracy plot
fig2_accuracy.update_layout(
    title='Training and Validation Accuracy',
    xaxis_title='Epoch',
    yaxis_title='Accuracy',
    legend_title='Legend',
    width=600,  # adjust size as needed
    height=400
)


# Show the figure
fig2_accuracy.show()

