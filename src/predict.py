import torch
import pickle
from torch.utils.data import DataLoader

from dataset import CombinedData
from all_models import CombinedNet
from metrics import Accuracy
import utils


BATCH_SIZE = 32

if torch.cuda.is_available():
    device=torch.device('cuda')
    print('running on gpu')
else:
    device=torch.device('cpu')
    print('running on cpu')

stories_test = utils.read_data('data/nlp2_test.csv')

embed_file_test = open("data/dictionary_commonsense_test.pickle",'rb')
embedding_test=pickle.load(embed_file_test)
embed_file_test.close()

test_dataloader = DataLoader(CombinedData(stories_test, embedding_test, device), 
                            batch_size=BATCH_SIZE, shuffle=False)

metric_acc = Accuracy()


with torch.no_grad():
    metric_acc.reset()

    model = CombinedNet(device, pretrained=(False, False, False))
    model.load_state_dict(torch.load('checkpoints/combined_model.pth'))
    model.to(device)
    for i, test_batch in enumerate(test_dataloader):
        logits = model(test_batch)

        _, predicted = torch.max(logits, 1)
        metric_acc.update_batch(predicted, test_batch['labels'])

    test_accuracy = metric_acc.get_metrics_summary()
    metric_acc.reset()

    print(f'======== TestAccuracy: {test_accuracy} ======')