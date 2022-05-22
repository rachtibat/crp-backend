from os import write
import torch
import torch
from pathlib import Path

from typing import Dict, List, Iterable, Tuple, Any, Union, Iterator
import numpy as np
import torch.utils.data
from CRP_backend.feature_visualization.utils import load_max_activation, load_receptive_field, load_rel_statistics
from CRP_backend.datatypes.data_utils import *
import concurrent.futures
from torch.multiprocessing import Pool, Process, Queue

from CRP_backend.datatypes.Extra_Dataset import Extra_Dataset

from torch.nn.functional import relu

import torchvision.transforms as T
import io


from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from cairosvg import svg2png



class SMILES_Dataset:

    def __init__(self, device, model_path, data_path, extra_paths: list):
        """

        Args:
            device: "cpu" or "cuda:integer" corresponding to torch.device
        """
        self.device = device

        print(f"Loading DMI {self.get_name_dataset()}...", end='')
        self.dataset = self.load_dataset(data_path)
        print(" finished.")
        self.extra_DIs = self.load_extra_dataset_objects(extra_paths)
        print("Loading model...", end='')
        self.model = self.build_model(model_path)
        print(" finished.")

    def get_name_dataset(self) -> str:
        """
        In order to enable caching of intermediate results as well as differentiating this dataset from
        extra datasets defined in <self.load_extra_dataset_objects>, a unqiue name must be given.
        """

        return "BaseDataset"

    def build_model(self, model_path=None) -> torch.nn.Module:
        """
        method initializes model (loading of weights, .eval(), shifting model to gpu etc.).
        Please make sure, that the last layer is not a softmax activation.

        Returns:
            torch.nn.Module without softmax activation at output
        """

        raise NotImplementedError

    def load_dataset(self, data_path=None) -> torch.utils.data.Dataset:
        """
        !Please do not preprocess the data in <self.load_dataset>. Preprocessing is done in
        <self.preprocess_data_batch>!

        Parameter:
            data_path: string path to data. You do not need to use it, if you hardcode it below.
        Returns:
             Map-style datasets see https://pytorch.org/docs/stable/data.html#iterable-style-datasets
        """

        raise NotImplementedError

    def preprocess_data_batch(self, data: torch.tensor) -> torch.tensor:
        """
        method implements preprocessing of data for the model.
        !Please do not preprocess the data in <self.load_dataset>!

        Parameter:
            data: (torch.tensor) input data for model

        Returns:
            torch.tensor
        """

        raise NotImplementedError

    def get_data_sample(self, index) -> Tuple[Any, Any]:
        """
        Interface describes how a single data sample is loaded from the dataset.
        If method not edited by user, returns single data sample using direct indexing of the dataset.

        Parameter:
            index value of data samples (integer)

        Returns:
            tuple, where first element is data sample at <index> (torch.tensor) and second element is the target.

        """

        data, target = self.dataset[index]
        return data.unsqueeze(0), target


    def define_canonizer(self) -> List:
        """
        method returns Zennit canonizers for the model if needed. BatchNorm Layer and Resnet Block require such
        a canonizer for example.
        """

        return None

    def load_extra_dataset_objects(self, data_paths: list) -> List[Extra_Dataset]:
        """
        This method enables to analyze concepts with an additional dataset that is not part of the training distribution.
        If no extra dataset is used, please return an empty list.

        Parameter:
            data_paths: list of (pathlib.Path object) path to data. You do not need to use it, if you hardcode it below.
        Returns:
            list of <Extra_Dataset> objects. Note: objects not classes!
        """

        return []

    def select_standard_target(self, multitarget):
        """
        Only relevant for multi-target settings or if sample exist without clear defined targets.
        If a multi-target dataset is used, <multitarget> contains for a sample several targets. For example 
        <multitarget> = [2, 5] for sample1 i.e. sample1's true class is "2" and "5".
        Since each explanation is w.r.t only ONE target, it must be specified which target should be used in a
        default setting. (The target can be dynamically changed in the frontend afterwards.)
        Default behavior: target defined and no mutli-target setting, thus return the single target as it is
        from <get_data_sample>.

        Parameter:
            pred: model prediction output with batch size equals 1
            target: second tuple of <get_data_sample> output
        Returns:
            single target i.e. for the example above "2" if first element is chosen per default. Or "max" if maximal
            output value of <pred> is selected (this behavior is implemented in <init_relevance_of_target>).
        """

        return multitarget
       

    def multitarget_to_single(self, multitarget) -> List:
        """
        for example:
        [1, 2, 3] -> [[1], [2], [3]]
        [0, 1, 0, 1] -> [[0, 1, 0, 0], [0, 0, 0, 1]]

        if emtpy element is returned, the corresponding sample is ignored
        i.e. [1, 2, 3] -> [[], [2], [3]] the first sample is not analyzed.

        Default behavior: not implemented and ignored
        """

        raise NotImplementedError


    def decode_pred(self, pred: torch.tensor, N: int) -> Tuple[List[str], List[float]]:
        """
        method returns an understandable representation for the user. Might be string values, images, audio etc.
        For now only string values are supported.

        Args:
            pred: (torch.tensor) model output
            N: N most confident predictions

        Returns:
            class names, confidence values as tuple
        """

        raise NotImplementedError

    def decode_target(self, target: torch.tensor) -> List[str]:
        """
        method returns an understandable representation for the user. Might be string values, images, audio etc.
        For now only string values are supported

        Args:
            target/label: (torch.tensor) used in loss function

        Returns:
            ?
        """

        raise NotImplementedError

    def decode_class_name(self, class_name:str):
        """
        decodes the user defined class names to a target used for heatmap calculations
        """

        raise NotImplementedError

    def get_all_targets(self):
        """
        method returns list of all possible targets
        Used if Relevance Maximization is w.r.t all classes computed.
        """

        raise NotImplementedError

    #TODO: program it
    def get_all_classes(self):
        """
        method returns all class labels
        """

        raise NotImplementedError

    def selection_of_indices_per_class(self) -> Dict[str, List[int]]:
        """
        This method is optional.
        If you like to select images using their class names instead of raw image indices values,
        you can specify this function to return a selection of indices for each class name.

        Returns:
        dictionary with keys are class names and value is list of index integers
        """

        return {}

    #TODO: only used in caching -> useless
    def data_index_to_filename(self, index: int) -> Union[str, int]:
        """
        the analysis results are saved for each data sample independently. Every data sample is identified with its index
        inside the dataset. If the ordering of the dataset is changed, the program can NOT find the cached results and
        wrong data is returned!
        To allow the user, to change the data ordering, he can define the method <data_index_to_name> and
        <data_name_to_index>. If the indices change, but the file names stay the same, no corruption of the analysis
        results occurs.
        Default behavior: file name is equal to index
        !Note, if extra datasets are used, please make sure that each data name is unique for all datasets!
        If filename not found, please raise an error in your implementation.

        Args:
            index: dataset index

        Returns:
            filename as string or integer

        """

        return index

    #TODO: only used in caching -> useless
    def data_filename_to_index(self, filename: Union[str, int]) -> int:
        """
        see <data_index_to_name>

        Args:
            filename as str or int

        Returns:
            index as integer
        """

        return int(filename)

    def pred_to_iterable(self, pred) -> Iterable:
        """
        method receives model output prediction and returns an "iterable" object, where each object
        is used to calculate the attribution inside the model.
        Zennit calculates the attribution by backpropagating gradients starting at each object. The gradients of each
        object are added together in each neuron.

        This method is useful, if the model produces several output objects and you either want to analyse all object or
        only a subset of them.

        Args:
            pred:

        Returns:
            iterable object

        """

        if type(pred) == torch.Tensor:
            return [pred]
        else:
            return pred

    def input_to_iterable(self, inputs) -> Iterable[torch.tensor]:
        """
        method receives data input and returns an "iterable" object, where each object
        is used to calculate the attribution inside the model.

        This method is useful, if the model has several inputs and you either want to see the heatmap of all inputs or
            only a subset of them.

        Args:
            pred:

        Returns:
            iterable object

        """

        if type(inputs) == torch.Tensor:
            return [inputs]
        else:
            return inputs

    def visualize_data_sample(self, image_np: np.ndarray, size: int) -> np.ndarray:
        """
        method receives first element of tuple output of self.get_data_sample and returns a numpy array on "cpu" device so that
        the image can be visualized.
        Args:
            data: self.get_data_sample[0]

        Returns:
            numpy array on cpu device
        """

        image_np = np.moveaxis(image_np, 0, -1)  # convert to channel last
        #loaded_image = convert_image_type(image_np, 0, 255, np.uint8)
        loaded_image = rescale_image(image_np, size)
        padded_image = pad_image(loaded_image)

        return padded_image

    def visualize_heatmap(self, relevance, size, encoded) -> np.ndarray:
        """
        method converts relevance output of <self.adjust_input_relevance> into a numpy array image
        Args:
            relevance:

        Returns:

        """

        raise NotImplementedError

    def visualize_synthetic(self, image_np: np.ndarray, size: int) -> np.ndarray:
        """
        method converts synthetic output of lucent into a numpy array image

        """
        image_np = np.moveaxis(image_np, 0, -1)  # convert to channel last
        image_np = convert_image_type(image_np, 0, 255, np.uint8)
        image_np = rescale_image(image_np, size)

        return image_np

    def adjust_input_relevance(self, inputs: List[torch.tensor]) -> torch.tensor:
        """
        method receives list of input relevances as torch.tensors and applies optional transformations on the generated
        attribution.
        For example, it is good practice to add up all channels in heatmap images if you are not interested in the
        detailed distribution of the relevances between each image channel.

        Args:
            inputs:

        Returns:

        """

        # remove channel dimension for single image, keep batch dimension
        return inputs[0].sum(2)


    def init_relevance_of_target(self, targets, pred) -> Iterable[torch.tensor]:
        """
        method is used to initialize the relevence in zennit if the attribution should be analyzed
        with respect to the target/label
        If element in target is "max", then the maximal prediction output is set.
        Args:
            targets: batch/list of several self.get_data_sample[1] or "max" string elements
            pred: batch/list of model predictions of self.get_data_sample[0]

        Returns:
            same shape and datatype as self.pred_to_iterable
        """

        r = torch.zeros_like(pred).to(self.device)
        max_args = torch.argmax(pred, dim=-1)

        for i, t in enumerate(targets):
            if type(t) == int:
                r[i, t] = 1
            elif t == "max":
                r[i, max_args[i]] = 1
            else:
                raise ValueError(f"target has wrong value with {t}")

        return [r]

    def get_synthetic_transform(self) -> Tuple[List, int]:
        
        standard_transform = [
            T.Pad(14, 0.5),
            T.RandomAffine(10, (0.0625, 0.0625), (0.9, 1.1))
        ]

        return standard_transform, 200

    def get_only_layers_to_analyze(self) -> List[str]:
        """
        this method returns a list of string layer names that specify which layers must be analyzed.
        Note that ONLY these layers are analyzed. If None, all possible layers are analyzed.

        layer names must be present int torch's <model.named_modules()>
        """
        return None

    def generate_input_attr_mask(self, inp_image, x, y, width, height):
        """
        method is used in a local analysis. It describes how the mask for <extract_rel_in_region> is generated.
        Args:
            coordinates: 0 < x,y,width, height < 1, normed according to image size

        Returns:

        """
        inp_shape = inp_image.shape
        x, y = int(x * inp_shape[3]), int(y * inp_shape[2])
        width, height = int(width * inp_shape[3]), int(height * inp_shape[2])

        mask = np.zeros(inp_shape[-2:])
        mask[y:y + height, x:x + width] = 1

        return mask

    def extract_rel_in_region(self, inp_attr_ch: np.ndarray, mask: np.ndarray):
        """
        method describes in a local analysis, how the mask is used to extract local relevance from the input
        attributions generated through self.adjust_input_relevance method.

        Args:
            inp_attr_ch: input attribution for several channel. length(inp_attr_ch) == number of channels to analyze.
            mask: output of <generate_input_attr_mask>

        Returns:
            array with length(inp_attr_ch)


        """

        rel_in_mask = inp_attr_ch * mask[None, ...]
        return np.sum(rel_in_mask, axis=(1, 2))

  
    def gen_visualize_filter(self, codes, most_neuron_index, rf, rf_to_neuron_index, filter_index, size, selected=(0, 10)):

        neuron_indices = most_neuron_index[selected[0]:selected[1], filter_index].astype(int)
        rf_indices = [rf_to_neuron_index[index] for index in neuron_indices]
        rf_n = rf[rf_indices]

        images_list = []
        for i_c, cod in enumerate(codes):
            # convert encoded smiles string back to human readable form
            # then convert to Chem.Mol and plot with SimilarityMaps.
            # Heatmap plot considers only the atoms and not the connections like "="
            # thus, we take only the relevance of the atoms and not "=" or "(" and so on.
            smiles_str = ""
            rf_on_letter = []

            for i, e in enumerate(cod.cpu().numpy()):
                letter = self.idx2token[e]
                smiles_str += self.idx2token[e]
               
                if letter.isalpha():
                    rf_on_letter.append(rf_n[i_c, i])
    
            mol = Chem.MolFromSmiles(smiles_str)
            hetatms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            past_i = 0
            atom_nr = []
            for i_atom, atom in enumerate(hetatms):
                rf_this_atom = np.sum(rf_on_letter[past_i:past_i+len(atom)])
                past_i += len(atom)
                if rf_this_atom > 30:
                    atom_nr.append(i_atom)

            mol = Chem.MolFromSmiles(smiles_str)
            rdDepictor.Compute2DCoords(mol)
            Chem.Kekulize(mol)
            if size == -1:
                drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
            else:
                drawer = rdMolDraw2D.MolDraw2DSVG(size, size)

            drawer.DrawMolecule(mol, highlightAtoms=atom_nr)

            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()

            with io.BytesIO() as buff:
                svg2png(bytestring=svg, write_to=buff)
                buff.seek(0)
                data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
                buff.close()

            img = cv2.imdecode(data, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images_list.append(img)

        return images_list


    #TODO: remove torch and put to numpy CPU?
    def mask_images_with_rf(self, rf: torch.Tensor, images: np.ndarray, threshold=40) -> list:
        """
        functions multiplies the receptive field with the input image and 
        the region outside of the receptive field (elements with zero value) are cut out.

        Parameters:
            rf : torch.tensor (receptive field) [batch, height, width]
            images : torch.tensor [batch, channels, height, width]

        Returns:
            cropped images as list of numpy arrays [batch, channels, individual height, individual width]
        """
        cropped_img = []

        # crop images
        for i in range(len(rf)):
            rows, columns = torch.where(rf[i] > threshold)
            row1 = rows.min() if len(rows) != 0 else 0
            row2 = rows.max() if len(rows) != 0 else -1
            col1 = columns.min() if len(columns) != 0 else 0
            col2 = columns.max() if len(columns) != 0 else -1

            if (row1 < row2) and (col1 < col2):
                cropped = images[i, ..., row1:row2, col1:col2]
            else:
                cropped = images[i]

            cropped_img.append(cropped)

        return cropped_img

    #TODO: delete
    def get_data_concurrently(self, indices: Union[List, np.ndarray, torch.tensor]):

        if len(indices) == 1:
            data, label = self.get_data_sample(indices[0])
            return data, label

        threads = []
        data_returned = []
        labels_returned = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for index in indices:
                future = executor.submit(self.get_data_sample, index)
                threads.append(future)

        for t in threads:
            single_data = t.result()[0]
            single_label = t.result()[1]
            data_returned.append(single_data)
            labels_returned.append(single_label)

        data_returned = torch.cat(data_returned, dim=0)
        return data_returned, labels_returned
