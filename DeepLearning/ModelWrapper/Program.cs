using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

using Python.Runtime;

namespace ModelWrapper
{
    class Program
    {
        static void Main(string[] args)
        {
            //model1
            string pythonPath = @"C:\ProgramData\Anaconda3\Lib\site-packages";
            string pythonCode = @"..\..\..\TestKeras.py";
            string modelFile = @"..\..\..\MNIST\keras.h5";
            //model2
            string pythonServiceUrl = "http://mircean-p710:1234";

            string input = @"C:\Users\mircean\git\DeepLearning\MNIST\test.csv";
            string output = @"C:\Users\mircean\git\DeepLearning\MNIST\predict.csv";

            int method = 2;
            if (method == 1)
            {
                ModelWrapper model = new ModelWrapper(pythonPath,
                    pythonCode,
                    modelFile);

                model.Predict(input, output);
            }
            if (method == 2)
            {
                ModelWrapper2 model = new ModelWrapper2(pythonServiceUrl);

                model.Predict(input, output);
            }
            using (StreamReader sr = new StreamReader(output))
            {
                var classId = sr.ReadLine();
                Debug.Assert(classId == "2");
            }
        }
    }

    class ModelWrapper
    {
        string m_pythonCode;
        string m_modelFile;

        public ModelWrapper(string pythonPath,
            string pythonCode,
            string modelFile)
        {
            FileInfo fileInfo = new FileInfo(pythonCode);

            using (Py.GIL())
            {
                dynamic py_sys = Py.Import("sys");
                py_sys.path.append(pythonPath);
                py_sys.path.append(fileInfo.Directory.FullName);
            }

            m_pythonCode = Path.GetFileNameWithoutExtension(fileInfo.Name);
            m_modelFile = modelFile;
        }

        public void Predict(string input, string output)
        {
            using (Py.GIL())
            {
                dynamic py_model = Py.Import(m_pythonCode);
                py_model.predict(m_modelFile, input, output);
            }
        }
    }

    class ModelWrapper2
    {
        Uri m_uri;
        static HttpClient client = new HttpClient();

        public ModelWrapper2(string url)
        {
            m_uri = new Uri(url);
        }

        public void Predict(string input, string output)
        {
            string requestString = "Body\r\n" + input + "\r\n" + output;
            var response = client.PostAsync(m_uri, new StringContent(requestString)).Result;
            Debug.Assert(response.StatusCode == HttpStatusCode.OK);
        }
    }
}
