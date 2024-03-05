import pandas as pd
from collections import defaultdict

def compute_score(pred:str, label:str):
    """
    Compute the score of the prediction
    :param pred: prediction
    :param label: ground truth
    :return: score
    """
    total_ones = label.count('1')
    correct_ones = sum(p == l == '1' for p, l in zip(pred, label))

    if any(p == '1' and l == '0' for p, l in zip(pred, label)):
        return 0
    elif total_ones == 0:
        return 0
    else:
        return correct_ones / total_ones

def compute_accuracy(pred:str, label:str):
    """
    Compute the accuracy of the prediction
    :param pred: prediction
    :param label: ground truth
    :return: accuracy
    """
    return 1 if pred == label else 0
    
# print(compute_score('10100', '11111'))

def submission_stat(submision_id:int, level:int=None):
    data_path = '../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/'
    public_test = pd.read_csv(data_path + 'labeled_public_test.csv', dtype={'label': str})
    submission = pd.read_csv(f'../submissions/submission_{submision_id}.csv', dtype={'answer': str})

    # score
    preds, labels = submission['answer'].tolist(), public_test['label'].tolist()
    scores = [compute_score(pred, label) for pred, label in zip(preds, labels)]
    final_score = sum(scores) / len(scores)

    # accuracy
    accuracies = [compute_accuracy(pred, label)*len(pred) for pred, label in zip(preds, labels)]
    final_accuracy = sum(accuracies) / sum([len(pred) for pred in preds])
    print(f'submission {submision_id}. score: {final_score:.3f}, accuracy: {final_accuracy:.8f}')
    # print('total options: ', sum([len(pred) for pred in preds]))

    # print incorrect predictions
    count = 0
    for i, row in submission.iterrows():
        if row['answer'] != public_test['label'][i]:
            if level is not None and row['id'][5] != str(level):
                continue
            print(f'index: {i}, id: {public_test["id"][i]}, prediction: {row["answer"]}, label: {public_test["label"][i]}')
            count += 1
    print(f'incorrect predictions: {count}')
    

# submission_stat([14])

def diff_predictions(sub1, sub2):
    print(f'submission {sub1} vs submission {sub2}')

    submission1 = pd.read_csv(f'../submissions/submission_{sub1}.csv', dtype={'answer': str})
    submission2 = pd.read_csv(f'../submissions/submission_{sub2}.csv', dtype={'answer': str})

    count = 0
    for i, row in submission1.iterrows():
        if row['answer'] != submission2['answer'][i]:
            print(f'index: {i}, id: {submission1["id"][i]}, prediction1: {row["answer"]}, prediction2: {submission2["answer"][i]}')
            count += 1
    print(f'mismatched predictions: {count}')

def raw_accuracy(i):
    data_path = '../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/'
    public_test = pd.read_csv(data_path + 'labeled_public_test.csv', dtype={'label': str})
    submission = pd.read_csv(f'../submissions/submission_{i}.csv', dtype={'answer': str})

    # accuracy
    preds, labels = submission['answer'].tolist(), public_test['label'].tolist()
    preds = sum([list(pred) for pred in preds], [])
    labels = sum([list(label) for label in labels], [])
    accuracies = [compute_accuracy(pred, label) for pred, label in zip(preds, labels)]
    final_accuracy = sum(accuracies) / len(accuracies)
    return final_accuracy

def accuracy_wrt_level(i):
    data_path = '../datasets/KALAPA_ByteBattles_2023_MEDICAL_Set1/'
    public_test = pd.read_csv(data_path + 'labeled_public_test.csv', dtype={'label': str})
    submission = pd.read_csv(f'../submissions/submission_{i}.csv', dtype={'answer': str})

    # accuracy
    preds, labels, levels = submission['answer'].tolist(), public_test['label'].tolist(), public_test['id'].tolist()

    levels = sum([[p[5]]*len(pred) for p, pred in zip(levels, preds)], [])
    preds = sum([list(pred) for pred in preds], [])
    labels = sum([list(label) for label in labels], [])
    accuracies = [compute_accuracy(pred, label) for pred, label in zip(preds, labels)]
    
    acc_dict = defaultdict(lambda: defaultdict(int))
    for acc, lvl in zip(accuracies, levels):
        acc_dict[lvl]['total'] += 1
        acc_dict[lvl]['correct'] += acc
    for lvl in sorted(acc_dict.keys()):
        acc_dict[lvl]['accuracy'] = acc_dict[lvl]['correct'] / acc_dict[lvl]['total']
        print(f"level: {lvl}, accuracy: {acc_dict[lvl]['correct']}/{acc_dict[lvl]['total']} ({acc_dict[lvl]['accuracy']*100:.2f} %)")        

# print(raw_accuracy(17))
submission_stat(11, 2)
# accuracy_wrt_level(11)