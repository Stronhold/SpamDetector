package modelgenerator;

import org.apache.avro.generic.GenericData;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.math.NumberUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static modelgenerator.Constants.*;

/**
 * Created by Sergio on 07/05/2016.
 */
public class ModelGenerator implements Serializable{

    public ModelGenerator(){
        //Eliminar modelo si existe
        File f = new File(Constants.SAVE_PATH);
        if(f.exists()) {
            try {
                FileUtils.deleteDirectory(new File(Constants.SAVE_PATH));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        //generamos la configuracion y el contexto
        SparkConf sparkConf = new SparkConf().setAppName("ModelGeneratorSMSSpam").setMaster("local[*]");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        JavaRDD<String> sms = jsc.textFile(Constants.SMS_PATH);
        //Leemos las stopwords
        JavaRDD<String> stopwords = jsc.textFile(Constants.STOPWORDS_PATH);
        //Obtenemos las frecuencias de cuántas veces aparece cada palabra
        JavaRDD<Tuple2<String, Double>> smsSpam = GetFrequencies(stopwords, sms, "spam\t", false, null);
        //obtenemos las 20 palabras más usadas
        List<String> list = smsSpam.map(new Function<Tuple2<String, Double>, String>() {
            public String call(Tuple2<String, Double> stringDoubleTuple2) throws Exception {
                return stringDoubleTuple2._1;
            }
        }).take(20);
        //Lo convertirmos a rdd
        JavaRDD<String> filtering = jsc.parallelize(list);
        //Obtenemos las frecuencias de cuántas veces aparece cada palabra
        JavaRDD<Tuple2<String, Double>> smsHarm = GetFrequencies(stopwords, sms, "ham\t", true, filtering);
        //Hacemos que las listas sean iguales (de 20 cada una), si una palabra no sale, la asignamos probabilidad de 0
        List<Tuple2<String, Double>> listaHarm = smsHarm.take(20);
        List<Tuple2<String, Double>> listaSpam = smsSpam.take(20);
        for(int i = 0; i < listaSpam.size(); i++){
            String name = listaSpam.get(i)._1;
            boolean found = false;
            for(Tuple2<String, Double> t: listaHarm){
                if(t._1.equals(name)){
                    found = true;
                    break;
                }
            }
            if(!found){
                listaHarm.add(new Tuple2<String, Double>(name, 0.0));
            }
        }
        //ordenamos las listas
        listaHarm.sort(new Comparator<Tuple2<String, Double>>() {
            public int compare(Tuple2<String, Double> o1, Tuple2<String, Double> o2) {
                return o1._1.compareTo(o2._1);
            }
        });
        listaSpam.sort(new Comparator<Tuple2<String, Double>>() {
            public int compare(Tuple2<String, Double> o1, Tuple2<String, Double> o2) {
                return o1._1.compareTo(o2._1);
            }
        });

        //Generamos los labeledpoints que se usan para NaiveBayes y les asignamos un numero
        List<LabeledPoint> aLabeled = new ArrayList<LabeledPoint>();
        int i = 0;
        double [] v = new double[listaHarm.size()];
        for(Tuple2<String, Double> t : listaHarm){
            v[i] = t._2;
            i++;
        }
        LabeledPoint l = new LabeledPoint(0.0, Vectors.dense(v));
        i = 0;
        aLabeled.add(l);
        for(Tuple2<String, Double> t : listaSpam){
            v[i] = t._2;
            i++;
        }
        LabeledPoint l2 = new LabeledPoint(1.0, Vectors.dense(v));
        aLabeled.add(l2);
        JavaRDD<LabeledPoint> labels = jsc.parallelize(aLabeled);

        //Generamos el modelo
        NaiveBayesModel model = NaiveBayes.train(labels.rdd());
        //Guardamos el modelo
        model.save(jsc.sc(), Constants.SAVE_PATH);
    }

    /**
     * Obtiene las frecuencias
     * @param stopwords stopwords
     * @param sms sms a analizar
     * @param filter filtro
     * @param intersects rdd con el que hacer interseccion
     * @param intersection indica si hacer una interseccion
     * @return
     */
    private JavaRDD<Tuple2<String, Double>> GetFrequencies(JavaRDD<String> stopwords, JavaRDD<String> sms, final String
            filter, boolean intersects, JavaRDD<String> intersection) {
        //realizacion de filtro
        sms = doFilter(stopwords, sms, filter);
        // Split words
        JavaRDD<String> smsWords = sms.flatMap(new FlatMapFunction<String, String>() {
            public Iterable<String> call(String s) throws Exception {
                return Arrays.asList(s.split(" "));
            }
        });
        //Eliminamos las stopwords
        smsWords = smsWords.subtract(stopwords);
        final int total = (int) smsWords.count();
        JavaPairRDD<String, Double> wordCount = null;
        //Si intersects es true,
        if(intersects == true) {
            // En este caso hacemos primero el mapeo para ver las caracteristicas que tenemos en cuenta
            // hacemos una interseccion con las palabras "buenas"
            smsWords = mapWordsNotPair(smsWords).intersection(intersection);
            //Hacemos un conteo de palabras
            wordCount = smsWords.mapToPair(new PairFunction<String, String, Double>() {
                public Tuple2<String, Double> call(String s) throws Exception {
                    return new Tuple2<String, Double>(s, 1d);
                }
            }).reduceByKey(new Function2<Double, Double, Double>() {
                public Double call(Double aDouble, Double aDouble2) throws Exception {
                    return aDouble + aDouble2;
                }
            });
        }
        else {
            //Contamos las palabras en funcion de varios parametros
            wordCount = doWordCount(smsWords);
        }
        //Obtenemos porcentajes
        JavaRDD<Tuple2<String, Double>> rdd = wordCount.map(new Function<Tuple2<String,Double>, Tuple2<String, Double>>() {
            public Tuple2<String, Double> call(Tuple2<String, Double> stringDoubleTuple2) throws Exception {
                return new Tuple2<String, Double>(stringDoubleTuple2._1, stringDoubleTuple2._2/total);
            }
        });

        return rdd.sortBy(new Function<Tuple2<String,Double>, Double>() {
            public Double call(Tuple2<String, Double> stringDoubleTuple2) throws Exception {
                return stringDoubleTuple2._2;
            }
        }, false, 1);
    }

    /**
     * Hacemos el conteo de palabras y dependiendo de las caracteristicas de las palabras
     * cambiamos su valor
     * @param smsWords rdd de palabras
     * @return
     */
    private JavaPairRDD<String, Double> doWordCount(JavaRDD<String> smsWords) {
        return mapWords(smsWords).reduceByKey(new Function2<Double, Double, Double>() {
            public Double call(Double a, Double b) { return a + b; }
        });
    }

    private JavaRDD<String> mapWordsNotPair(JavaRDD<String> smsWords){
        return smsWords.map(new Function<String, String>() {
            public String call(String s) throws Exception {
                s = s.replace("?", "").replace(":","").replace("?","").replace("!", "").replace(",", "");
                if(s.endsWith(".")){
                    s = s.substring(0, s.length() - 1);
                }
                else if(s.contains("%")){
                    return PERCENTAGES;
                }
                else if(isPhoneNumberValid(s)){
                    return PHONENUMBERS;
                }
                else if(s.contains("$") || s.contains("€") || s.contains("£")){
                    return MONEYSYMBOL;
                }
                else if(NumberUtils.isNumber(s) || s.matches(".*\\d+.*")){
                    return Constants.NUMBERS;
                }
                //check if capitals
                else if(s.equals(s.toUpperCase())){
                    return Constants.CAPITALS;
                }
                return s.toLowerCase();
            }
        });
    }
    private JavaPairRDD mapWords(JavaRDD<String> smsWords) {
        return smsWords.mapToPair(new PairFunction<String, String, Double>() {
            public Tuple2<String, Double> call(String s) throws Exception {
                s = s.replace("?", "").replace(":","").replace("?","").replace("!", "").replace(",", "");
                if(s.endsWith(".")){
                    s = s.substring(0, s.length() - 1);
                }
                else if(s.contains("%")){
                    return new Tuple2<String, Double>(PERCENTAGES, (double) 1);
                }
                else if(isPhoneNumberValid(s)){
                    return new Tuple2<String, Double>(PHONENUMBERS, (double) 1);
                }
                else if(s.contains("$") || s.contains("€") || s.contains("£")){
                    return new Tuple2<String, Double>(MONEYSYMBOL, (double) 1);
                }
                else if(NumberUtils.isNumber(s) || s.matches(".*\\d+.*")){
                    return new Tuple2<String, Double>(Constants.NUMBERS, (double) 1);
                }
                //check if capitals
                else if(s.equals(s.toUpperCase())){
                    return new Tuple2<String, Double>(Constants.CAPITALS, (double) 1);
                }
                return new Tuple2<String, Double>(s.toLowerCase(), (double) 1);
            }
        });
    }

    /**
     * Hacemos el filtro
     * @param stopwords stopwords
     * @param sms lista de sms
     * @param filter por qué debemos filtrar
     * @return
     */
    private JavaRDD<String> doFilter(JavaRDD<String> stopwords, JavaRDD<String> sms, final String filter) {
        sms = sms.filter(new Function<String, Boolean>() {
            public Boolean call(String s) throws Exception {
                if(s.contains(filter)){
                    return true;
                }
                return false;
            }
        });
        //remove filter
        return sms.map(new Function<String, String>() {

            public String call(String s) throws Exception {
                return s.replace(filter, "");
            }
        });
    }

    /**
     * Comprueba si es un numero de telefono
     * @param phoneNumber tfno
     * @return
     */
    public boolean isPhoneNumberValid(String phoneNumber) {
        if (!phoneNumber.matches("^(([(]?(\\d{2,4})[)]?)|(\\d{2,4})|([+1-9]+\\d{1,2}))?[-\\s]?(\\d{2,3})?[-\\s]?(" +
                "(\\d{7,8})|(\\d{3,4}[-\\s]\\d{3,4}))$\n")) {
            return false;
        }
        return true;
    }
}
