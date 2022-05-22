## run tests
import warnings
import torch
from CRP_backend.feature_visualization.GradientAscent import GradientAscent
import matplotlib.pyplot as plt
from CRP_backend.datatypes.Image_2D_Dataset import Image_2D_Dataset

from experiments.VGG16_Adience_no_bn.load_dataset import preprocessing

class TestSuite:

    def __init__(self, SDS):

        self.DMI = SDS.DMI
        self.SDS = SDS

    def run_all_tests(self):

        for i, DI in enumerate(self.SDS.all_DIs):

            print(f"# Test dataset {DI.get_name_dataset()} ... ")

            self.test_dataset(DI, i)
            self.test_model_inference(DI, i)
            self.test_decode_pred(DI, i)
            self.test_data_index_name(DI)

            print("successful!")     

        print("# Test DMI ...")
        self.test_softmax_layer()
        self.test_decode_target()
        self.test_to_iterable()
        self.test_zennit()
        print("All tests completed!")    


    def test_softmax_layer(self):

        for name, module in self.DMI.model.named_modules():

            if isinstance(module, torch.nn.Softmax):
                warnings.warn("Softmax Layer found in Model! LRP is not working well with Softmax outputs.")

    def test_dataset(self, DI, i):
        """
        if i == 0, base dataset
        """
        print(f"Test dataset attributes ... ", end="")

        if not hasattr(DI.dataset, "__getitem__") or not hasattr(DI.dataset, "__len__"):
            raise ValueError("Dataset must implement __getitem__() and __len__()")

        if i == 0:

            data_sample = DI.get_data_sample(0)
            if len(data_sample) != 2:
                raise ValueError(
                    "get_data_sample must return a tuple with length 2, where the first element is the model input"
                    "data and second element is the target.")

        else:

            data_sample = DI.get_data_sample_no_target(0)
            if len(data_sample) != 1:
                raise ValueError("get_data_sample_no_target must a single value, no tuple with a target")

        print(" successful!")

        

    def test_model_inference(self, DI, i):

        print("Test model inference. Run <data, label = get_data_sample(0)> and feed output to <model(data)>...", end="")

        if i == 0:
            data_sample, _ = DI.get_data_sample(0, preprocessing=True)
        else:
            data_sample = DI.get_data_sample_no_target(0, preprocessing=True)

        #data_sample = self.DMI.preprocess_data_batch(data_sample)
        _ = self.DMI.model(data_sample)

        print(" successful.")

    def test_to_iterable(self):
        print("Test <input_to_iterable>, <pred_to_iterable> and <init_relevance_of_target> in DataModelInterface...", end="")

        data_sample, target = self.DMI.get_data_sample(0)
        data_list = self.DMI.input_to_iterable(data_sample)

        for d in data_list:

            if type(d) != torch.Tensor:
                raise ValueError("Elements of the return value of <input_to_iterable> must be torch.tensor.")

        pred = self.DMI.model(data_sample)

        pred_list = self.DMI.pred_to_iterable(pred)
        rel_list = self.DMI.init_relevance_of_target([target], pred)

        if len(pred_list) != len(rel_list):
            raise ValueError("Output of <pred_to_iterable> must have same length as <init_relevance_of_target>.")

        for p, r in zip(pred_list, rel_list):

            if type(p) != torch.Tensor:
                raise ValueError("Elements of the return value of <pred_to_iterable> must be torch.tensor.")

            if type(r) != torch.Tensor:
                raise ValueError("Elements of the return value of <init_relevance_of_target> must be torch.tensor.")

            if r.shape != p.shape:
                raise ValueError("Shape of model prediction must be same as relevance initialization.")

        print(" successful.")

    def test_data_index_name(self, DI):

        print("Test <data_index_to_filename> and <data_filename_to_index>...", end="")

        filename = DI.data_index_to_filename(0)
        index = DI.data_filename_to_index(filename)

        if 0 != index:
            raise ValueError("Error in <data_index_to_filename> and <data_filename_to_index>."
                             "index and filename must be invertible, adhere to a one-to-one correspondence")

        print(" successful.")

    def test_decode_target(self):

        print("Test <decode_target>...", end="")

        _, label = self.DMI.get_data_sample(0)

        try:
            single_targets = self.DMI.multitarget_to_single(label)
            if len(single_targets) == 0:
                print(f"target {single_targets} is empty, datapoint will be ignored in analysis")
            else:
                for t in single_targets:
                    decoded = self.DMI.decode_target(t)
                    print(f" For target {t} returns {decoded}")
        except NotImplementedError:
            print("No multi-target behavior defined in <multitarget_to_single>")
            decoded = self.DMI.decode_target(label)
            print(f" For target {label} returns {decoded}")

        standard_t = self.DMI.select_standard_target(label)
        decoded = self.DMI.decode_target(standard_t)
        print(f"For target {label} returns standard target defined in <select_standard_target> as {decoded}")

        
    def test_decode_pred(self, DI, i):


        if i == 0:
            data_sample, _ = DI.get_data_sample(0, preprocessing=True)
        else:
            return 

        print("Test <decode_pred>...", end="")

        pred = self.DMI.model(data_sample)
        decoded = self.DMI.decode_pred(pred, 1)

        print(f" For prediction {pred} returns {decoded}")

    #TODO: implemenet
    def test_zennit(self):

        pass

    #TODO: delete
    def generate_synthetic_images(self, exp_name, MG):

        if not issubclass(self.DMI.__class__, Image_2D_Dataset):
            warnings.warn("You dare not using ImageDataset. Generation of synthetic images is only for ImageDataset possible.")
            return

        print("Test input shape for generation of synthetic images...", end="")

        data_sample, _ = self.DMI.get_data_sample(0)

        if len(data_sample.shape[1:]) != 3:

            warnings.warn("Generation of synthetic images is for this data shape not possible!")
            return

        print(" completed.")

        #TODO: delete
        # print("Generate synthetic images using transforms, steps, layer names and channels of <test_lucent_transforms> ...", end="")

        # test_transforms, steps, layer, channel = self.DMI.test_lucent_transforms()

        # if len(layer) == 0:
        #     model_layers = list(MG.named_modules.keys())
        #     n_layers = len(model_layers)
        #     layer = [model_layers[n_layers//2]]

        # if len(channel) == 0:
        #     channel = [0]

        # n_steps = len(steps)
        # n_transforms = len(test_transforms)

        # for l in layer:
        #     for c in channel:
        #         k = 1
        #         plt.figure()
        #         for i_t, t in enumerate(test_transforms):
        #             for s in steps:

        #                 GA = GradientAscent(exp_name, MG ,self.DMI, self.DMI.device)
        #                 GA.lucent_transforms = t
        #                 GA.lucent_steps = s
        #                 lucent_name = GA.layer_name_to_lucid[l]
        #                 image = GA.gen_synthetic_images(lucent_name, [c])

        #                 plt.subplot(n_transforms, n_steps, k)
        #                 plt.imshow(image[0])
        #                 plt.xticks([])
        #                 plt.yticks([])
        #                 plt.title(f"transform {i_t} steps {s}")
        #                 k += 1

        #         plt.savefig(f"layer_{l}_channel_{c}.png")

        # print(" completed.")



#TODO: delete
class TestMG():

    def __init__(self, DMI, MG) -> None:

        self.MG = MG
        self.DMI = DMI

    def test_only_layers_to_analyze(self):

        names = self.DMI.get_only_layers_to_analyze()

        for name in names:
            
            layer = self.MG.named_modules[name]




    

