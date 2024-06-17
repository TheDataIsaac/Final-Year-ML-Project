import pickle
import pandas as pd
import numpy as np
from matminer.datasets import load_dataset
from pymatgen.core.composition import Composition
from matminer.featurizers.composition import ElementProperty, ElementFraction
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, f1_score, precision_score, recall_score

class DataLoader:
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def load_data(self):
        data = load_dataset("superconductivity2018")['composition'].to_frame().iloc[:self.num_samples]
        
        ef_ftd = pd.read_csv("df_sc_ef_ElementFraction_ftd.csv").dropna().reset_index()
        ep_ftd = pd.read_csv("df_sc_ep_ElementProperty_Magpie_ftd.csv").dropna().reset_index()
        
        ef_ftd = ef_ftd.drop_duplicates(subset=["Critical Temp", "_Composition"])
        ep_ftd = ep_ftd.drop_duplicates(subset=["Critical Temp", "_Composition"])

        return data, ep_ftd, ef_ftd

    def prepare_compositions(self):
        data, ep_ftd, ef_ftd = self.load_data()
        data['_Composition'] = data['composition'].apply(self.create_composition).to_frame()
        data = data.dropna().drop_duplicates()
        return data, ep_ftd, ef_ftd

    @staticmethod
    def create_composition(formula):
        try:
            return Composition(formula)
        except ValueError:
            print(f"Error parsing formula: {formula}")
            return None

class FeatureExtractor:
    def featurize(self, data):
        data = data.dropna()
        ep_featurizer = ElementProperty.from_preset('magpie')
        ep_ftd = ep_featurizer.featurize_dataframe(data, col_id='_Composition', ignore_errors=True)
        
        ef_featurizer = ElementFraction()
        ef_ftd = ef_featurizer.featurize_dataframe(data, col_id='_Composition', ignore_errors=True)
        return ep_ftd, ef_ftd
    
class ModelManager:
    def __init__(self):
        self.models = {
            'model_ep': 'sc_ep_rf_cl.pkl',
            'model_efep': 'sc_efep_rf_cl.pkl',
            'model_ef': 'sc_ef_rf_cl.pkl',
            'regmodel_efep': 'sc_efep_best_model_yet.pkl'
        }
        
        for name, path in self.models.items():
            with open(path, 'rb') as file:
                self.models[name] = pickle.load(file)

    def predict(self, model_name, X):
        model = self.models.get(model_name)
        if model:
            return model.predict(X)
        else:
            raise ValueError(f"Model '{model_name}' not found")

    def evaluate(self, pred_type, y_true, y_pred):
        if pred_type == "C":
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "confusion_matrix": confusion_matrix(y_true, y_pred),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1_score": f1_score(y_true, y_pred),
                "roc_auc_score": roc_auc_score(y_true, y_pred)
            }
        elif pred_type == "R":
            return {
                "Mean Squared Error (MSE)": mean_squared_error(y_true, y_pred),
                "R-squared": r2_score(y_true, y_pred),
                "Mean Absolute Error (MAE)": mean_absolute_error(y_true, y_pred),
                "Root Mean Squared Error (RMSE)": np.sqrt(mean_squared_error(y_true, y_pred)),
                "Adjusted R-squared": r2_score(y_true, y_pred, multioutput='uniform_average')
            }

class SuperconductorPredictor:
    def __init__(self, data_loader, feature_extractor, model_manager):
        self.data, self.ep_ftd, self.ef_ftd = data_loader.prepare_compositions()
        self.feature_extractor = feature_extractor
        self.model_manager = model_manager
        self.final_ep_ftd, self.final_ef_ftd = self.record_indices()
        

    def record_indices(self):
        compositions = self.data["composition"].tolist()
        print("Composition:", len(compositions))
        present_indices = []
        missing_indices = []
        
        for idx, composition in enumerate(compositions):
            if composition in self.ep_ftd['composition'].tolist():
                found_index = self.ep_ftd[self.ep_ftd['composition'] == composition]["index"].tolist()
                present_indices.extend(found_index)
            else:
                missing_indices.append(idx)
        
        print("PRESENT INDICES", present_indices[-10:])
        print("MISSING INDICES", missing_indices)

        present_ep_ftd = pd.DataFrame()
        present_ef_ftd = pd.DataFrame()
        missing_ep_ftd = pd.DataFrame()
        missing_ef_ftd = pd.DataFrame()

        if present_indices:
            present_ep_ftd = self.ep_ftd.loc[self.ep_ftd["index"].isin(present_indices)]
            present_ef_ftd = self.ef_ftd.loc[self.ef_ftd["index"].isin(present_indices)]
         
        if missing_indices:
            missing_data = self.data.iloc[missing_indices]
            missing_ep_ftd, missing_ef_ftd = self.feature_extractor.featurize(missing_data)
        
        final_ep_ftd = pd.concat([present_ep_ftd, missing_ep_ftd], ignore_index=True)
        final_ef_ftd = pd.concat([present_ef_ftd, missing_ef_ftd], ignore_index=True)

        final_nn_ep_ftd = final_ep_ftd.dropna()
        final_nn_ef_ftd = final_ef_ftd.dropna()
        
        final_in_ep_ftd = final_ep_ftd[final_ep_ftd.isnull().any(axis=1)]["composition"].tolist()
        final_in_ef_ftd = final_ef_ftd[final_ef_ftd.isnull().any(axis=1)]["composition"].tolist()
        print("Compositions with Null Values (unsuitable for prediction):", final_in_ep_ftd)
        print("Compositions with Null Values (unsuitable for prediction):", final_in_ef_ftd)
        
        return final_nn_ep_ftd, final_nn_ef_ftd

    def preprocess_data(self, ep_ftd, ef_ftd):
        def preprocess_for_model1(ep_ftd):
            ep_ftd["having_tc"] = (ep_ftd["Critical Temp"] >= 10).astype(int)
            ep_X = ep_ftd.iloc[:, 4:-1]
            ep_y = ep_ftd['having_tc']
            return ep_X, ep_y

        def preprocess_for_model3(ef_ftd):
            ef_ftd["having_tc"] = (ef_ftd["Critical Temp"] >= 10).astype(int)
            ef_X = ef_ftd.iloc[:, 4:-1]
            ef_y = ef_ftd['having_tc']
            return ef_X, ef_y

        def preprocess_for_model2(ep_ftd, ef_ftd):
            ef_ftd = ef_ftd.iloc[:, 2:]
            ep_ftd = ep_ftd.iloc[:, 2:]
            
            merged_df = pd.merge(ef_ftd, ep_ftd, left_on=["Critical Temp", "_Composition"], right_on=["Critical Temp", "_Composition"], how="inner")
            merged_df["having_tc"] = (merged_df["Critical Temp"] >= 10).astype(int)

            efep_X = merged_df.iloc[:, 2:-1]
            efep_y = merged_df['having_tc']    
            return efep_X, efep_y, merged_df

        ep_X, ep_y = preprocess_for_model1(ep_ftd)
        efep_X, efep_y, _ = preprocess_for_model2(ep_ftd, ef_ftd)
        ef_X, ef_y = preprocess_for_model3(ef_ftd)
        
        return ep_X, ep_y, efep_X, efep_y, ef_X, ef_y

    def preprocess_for_regmodel(self, result_df):
        indices = self.result_df.index
        ef_ftd = self.ef_ftd.iloc[indices, 2:]
        ep_ftd = self.ep_ftd.iloc[indices, 2:]
        
        merged_df = pd.merge(ef_ftd, ep_ftd, left_on=["Critical Temp", "_Composition"], right_on=["Critical Temp", "_Composition"], how="inner")
    
        efep_X = merged_df.iloc[:, 2:]
        efep_y = merged_df['Critical Temp']    
        
        print(efep_X.shape)
        return efep_X, efep_y

    def ensemble_predict(self):
        ep_X, ep_y, efep_X, efep_y, ef_X, ef_y = self.preprocess_data(self.final_ep_ftd, self.final_ef_ftd)
        
        pred_ep = self.model_manager.predict('model_ep', ep_X)
        pred_efep = self.model_manager.predict('model_efep', efep_X)
        pred_ef = self.model_manager.predict('model_ef', ef_X)
        
        n_samples = len(pred_ep)
        ensemble_pred = np.zeros(n_samples)
        for i in range(n_samples):
            class_counts = np.bincount([pred_ep[i], pred_efep[i], pred_ef[i]])
            ensemble_pred[i] = np.argmax(class_counts)

        ensemble_pred = ensemble_pred.astype(int)
        return ensemble_pred, ep_y, efep_y, ef_y

    def reg_predict(self, result_df):
        efep_X, efep_y = self.preprocess_for_regmodel(result_df)
        efep_pred = self.model_manager.predict('regmodel_efep', efep_X)

        return efep_pred, efep_y

    def evaluate_ensemble(self, pred_type, ep_y, ensemble_pred):
        eval_results = self.model_manager.evaluate(pred_type, ep_y, ensemble_pred)
        return eval_results

    def create_result_dataframe(self, ensemble_pred):
        result_df = pd.DataFrame({
            "composition": self.final_ef_ftd["_Composition"],
            "Tc": self.final_ef_ftd["Critical Temp"],
            "prediction": ensemble_pred
        })
        result_df = result_df[result_df["prediction"] == 1]
        self.result_df = result_df.copy()
        print(result_df.index)
        
        return result_df

    def predict_single_element(self, formula):
        composition = Composition(formula)
        single_element_df = pd.DataFrame({'composition': [composition]})
        ep_ftd, ef_ftd = self.feature_extractor.featurize(single_element_df)
        ensemble_pred, _, _, _, _ = self.ensemble_predict(ep_ftd, ef_ftd)
        result_df = self.create_result_dataframe(ensemble_pred, ef_ftd)
        return result_df.iloc[0]

    def predict(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.drop_duplicates()
            ep_ftd, ef_ftd = self.feature_extractor.featurize(data)
            final_nn_ep_ftd, final_nn_ef_ftd, final_in_ep_ftd, final_in_ef_ftd = self.record_indices(ep_ftd, ef_ftd, data)
            ensemble_pred, ep_y, efep_y, ef_y, ef_ftd = self.ensemble_predict(final_nn_ep_ftd, final_nn_ef_ftd)
            result_df = self.create_result_dataframe(ensemble_pred, ef_ftd)
            return result_df
        elif isinstance(data, str):
            return self.predict_single_element(data)
        else:
            raise ValueError("Unsupported data type. Please provide a DataFrame or a chemical formula as a string.")
        

# Initialize the components
data_loader = DataLoader(100)
feature_extractor = FeatureExtractor()
model_manager = ModelManager()
predictor = SuperconductorPredictor(data_loader, feature_extractor, model_manager)

ensemble_pred, ep_y, efep_y, ef_y = predictor.ensemble_predict()
print(predictor.evaluate_ensemble("C", ep_y, ensemble_pred))

result_dataframe = predictor.create_result_dataframe(ensemble_pred)

efep_pred, efep_y = predictor.reg_predict(result_dataframe)
print(predictor.evaluate_ensemble("R", efep_y, efep_pred))


