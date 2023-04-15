def model_inference_one_vac(model, dataset, vac_text):
  '''
  Инференс на 1 вакансии. У вас функция будет отличаться, очевидно
  '''
  # проваливаемся в кластер по локации
  location = dataset[dataset['vac_des'] == vac_text]['City'].iloc[0]
  cluster = dataset[(dataset["City"] == location) | \
                    (dataset["microcat_name"] =="Удаленная работа") | \
                    (dataset['microcat_name'] == 'Вахтовый метод')]

  vac_idx = make_indexes_from_tuple(tuple([vac_text]), vac_vocab).to(DEVICE)

  similarities = []
  for res_text in cluster['res_des']:
    if res_text:
      res_idx = make_indexes_from_tuple(tuple([res_text]), res_vocab).to(DEVICE)

      similarity = model(vac_idx, res_idx)
      similarities.append(similarity.item())
    else:
      similarities.append(-1)

  return similarities, np.arange(cluster.shape[0])

def model_inference_dataset(model, dataset):
  '''
  Инференс на всех вакансиях в dataset
  '''
  preds = np.zeros((dataset.shape[0]))
  for vac in dataset['vac_des'].unique():
    model_pred, pred_ind = model_inference_one_vac(model, dataset, vac)
    preds[pred_ind] = model_pred

  return preds
