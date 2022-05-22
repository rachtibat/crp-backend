from pathlib import Path
import os
import xml.etree.ElementTree as ET

class ConceptNaming:

    def __init__(self, save_path) -> None:
        
        self.save_path = save_path / Path("ConceptNames")

    def add_suggestion(self, layer, filter_index, concept_name):

            if not self.save_path.exists():
                os.makedirs(self.save_path)

            try:

                xml_file = ET.parse(self.save_path / Path(f"{layer}.xml"))
                xml_file = xml_file.getroot()
            
            except FileNotFoundError as e:
            
                xml_file = ET.Element("root")

            element = xml_file.find(f"f_{filter_index}")
            if element is None:
                element = ET.SubElement(xml_file, f"f_{filter_index}")

            if element.text:
                element.text += ";" + concept_name
            else:
                element.text = concept_name
            
            tree = ET.ElementTree(xml_file)
            tree.write(self.save_path / Path(f"{layer}.xml"))


    def overwrite_concept_name(self, layer, filter_index, concept_name):

        if not self.save_path.exists():
            os.makedirs(self.save_path)

        try:

            xml_file = ET.parse(self.save_path / Path(f"{layer}.xml"))
            xml_file = xml_file.getroot()
        
        except FileNotFoundError as e:
        
            xml_file = ET.Element("root")

        element = xml_file.find(f"f_{filter_index}")
        if element is None:
            element = ET.SubElement(xml_file, f"f_{filter_index}")

        element.text = concept_name
        
        tree = ET.ElementTree(xml_file)
        tree.write(self.save_path / Path(f"{layer}.xml"))


    def get_concept_names(self, layer, filter_indices: list):

        names = {}
        try:

            xml_file = ET.parse(self.save_path / Path(f"{layer}.xml"))
            xml_file = xml_file.getroot()

        except FileNotFoundError as e:
            
            return names

        for index in filter_indices:

            element = xml_file.find(f"f_{index}")
            if element is not None:
                names[str(index)] = str(element.text)

        return names
            

        
if __name__ == "__main__":
        
    path = "/home/achtibat/PycharmProjects/CRP_backend/CRP_backend/experiments/LeNet/"
    CN = ConceptNaming(path)

    CN.add_suggestion("layer1", 21, "New Name")
    CN.add_suggestion("layer1", 22, "New 22")

    print(CN.get_concept_names("layer1", [33, 323, 22]))