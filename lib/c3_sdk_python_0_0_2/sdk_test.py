from ctypes import *
import unittest
import sdk
import json

hexutil = CDLL('./lib/hexutil.so')

c3 = sdk.NewC3(stateFilePath = './lib/state.json')

class TestSDK(unittest.TestCase):
    def test_registerAndInvokeMethod(self):
        key = ""
        val = ""
        expectKey = "expectKey"
        expectVal = "expectVal"
        methodName = "foo"

        inputKey = c_char_p(hexutil.EncodeString(c_char_p(expectKey.encode('utf-8')))).value.decode('utf-8')
        inputVal = c_char_p(hexutil.EncodeString(c_char_p(expectVal.encode('utf-8')))).value.decode('utf-8')

        def setStuff(k, v):
            nonlocal key
            key = k

            nonlocal val
            val = v

        c3.registerMethod(methodName, setStuff)
        c3.invoke(methodName, inputKey, inputVal)

        self.assertEqual(expectKey, key)
        self.assertEqual(expectVal, val)

    def test_store(self):
        key = "foo"
        val = "bar"

        c3.state[key] = val

        self.assertEqual(c3.state[key], val)
        del c3.state[key]

    def test_state(self):
        methodName = "setState"

        key1 = "foo"
        val1 = "bar"

        key2 = "foofoo"
        val2 = "barbar"

        def setState(k, v):
            c3.state[k] = v

        c3.registerMethod(methodName, setState)

        p1 = [
            methodName,
            c_char_p(hexutil.EncodeString(c_char_p(key1.encode('utf-8')))).value.decode('utf-8'),
            c_char_p(hexutil.EncodeString(c_char_p(val1.encode('utf-8')))).value.decode('utf-8'),
        ]
        p2 = [
            methodName,
            c_char_p(hexutil.EncodeString(c_char_p(key2.encode('utf-8')))).value.decode('utf-8'),
            c_char_p(hexutil.EncodeString(c_char_p(val2.encode('utf-8')))).value.decode('utf-8'),
        ]

        params = [p1, p2]
        paramsJSON = json.dumps(params)

        c3.process(paramsJSON)

        self.assertEqual(c3.state[key1], val1)
        self.assertEqual(c3.state[key2], val2)

if __name__ == '__main__':
    unittest.main()
