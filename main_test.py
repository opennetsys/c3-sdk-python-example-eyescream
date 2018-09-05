import unittest
import os
import main
from lib.c3_sdk_python_0_0_2 import sdk
from lib.eyescream.dataset import generate_dataset as gd
from PIL import Image

testDir = "test_files"
testAugRelPath = testDir + os.path.sep + "out_aug_64x64"
testUnaugRelPath = testDir + os.path.sep + "out_unaug_64x64"
testNetworkRelPath = testDir + os.path.sep + "network"
testInputRelPath = testDir + os.path.sep + "input"
testAbsPath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + testDir
testAugAbsPath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + testAugRelPath
testUnaugAbsPath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + testUnaugRelPath
testNetworkAbsPath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + testNetworkRelPath
testInputAbsPath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + testInputRelPath

class TestMain(unittest.TestCase):
    def test_gd_gen(self):
        gd.gen(testInputAbsPath, testAugAbsPath, testUnaugAbsPath)

        ds = gd.Dataset([testAugAbsPath])
        images = ds.get_images()
        self.assertTrue(0 < len(list(images)))

    def test_accept_image(self):
        main.c3 = sdk.NewC3()
        main.initState()

        path = testInputAbsPath + os.path.sep + "face.jpg"
        img = Image.open(path)
        main.acceptImage(img)

        self.assertTrue(0 < len(list(c3.state[main.networkKey])))
        self.assertTrue(0 < len(list(c3.state[main.augImagesKey])))

if __name__ == '__main__':
    unittest.main()
