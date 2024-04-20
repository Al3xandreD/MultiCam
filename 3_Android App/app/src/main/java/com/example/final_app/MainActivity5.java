//package com.example.final_app;
//
//import androidx.appcompat.app.AppCompatActivity;
//import android.content.Intent;
//import android.os.Bundle;
//import android.view.View;
//import android.view.ViewGroup;
//import android.widget.AdapterView;
//import android.widget.Toast;
//import com.example.final_app.databinding.Activity2Binding;
//import com.example.final_app.databinding.ActivityMain5Binding;
//
//import java.util.ArrayList;
//
//// todo : remplacer l'utilisation des etages par des batiments
//
//public class MainActivity5 extends AppCompatActivity {
//
//    MainActivity5 binding;
//    DataEtageAdapter dataEtageAdapter;
//    ArrayList<Batiment> batimentArrayList = new ArrayList<>();
//
//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        binding = Activity2Binding.inflate(getLayoutInflater());
//        setContentView(binding.getRoot());
//
//        Intent intent = this.getIntent();
//        Batiment batiment = intent.getParcelableExtra("current_batiment");
//
//        for (int i = 0; i < batiment.etages.length; i++) {
//            for (int j = 0; j < batiment.etages[i].cameras.length; j++) {
//                DataEtage dataEtage_inter = new DataEtage("Etage" + Integer.toString(i), "Camera" + Integer.toString(j), Integer.toString(batiment.etages[i].victimes), R.drawable.logo_carte, R.drawable.logo_camera);
//                batimentArrayList.add(dataEtage_inter);
//            }
//        }
//
//        dataEtageAdapter = new DataEtageAdapter(MainActivity5.this, dataEtageArrayList);
//        binding.ListViewActivity2.setAdapter(dataEtageAdapter);
//
//        binding.ListViewActivity2.setOnItemClickListener(new AdapterView.OnItemClickListener() {
//            @Override
//            public void onItemClick(AdapterView<?> adapterView, View view, int i, long l) {
//                // Obtenez l'élément de données à partir de la position cliquée
//                DataEtage clickedDataEtage = dataEtageArrayList.get(i);
//
//                // Créez une intention pour démarrer l'ActivityCamera avec des données supplémentaires
//                Intent intent_act2 = new Intent(MainActivity5.this, ActivityCamera.class);
//                // Ajoutez des données supplémentaires à l'intention
//                intent_act2.putExtra("clicked_data_etage", clickedDataEtage);
//                // Démarrez l'activité avec l'intention
//                startActivity(intent_act2);
//            }
//        });
//    }
//}
//
//
////package com.example.final_app;
////
////import android.content.Intent;
////import android.graphics.Color;
////import android.os.Bundle;
////import android.view.View;
////import android.widget.AdapterView;
////
////import androidx.appcompat.app.AppCompatActivity;
////
////import com.example.final_app.databinding.ActivityMain5Binding;
////
////import java.util.ArrayList;
////
////
////public class MainActivity5 extends AppCompatActivity {
////
////    Etage etages[];
////    Batiment bat1, bat2, bat3;
////
////    /////////// Utilisation de binding car sencé être mieux que de la faire avec de findviewById
////    ActivityMain5Binding binding;
////    EtageAdapter5 batimentAdapter;
////    ArrayList<Batiment> batimentArrayList = new ArrayList<>();
////    Batiment batiment;
////
////
////    @Override
////    protected void onCreate(Bundle savedInstanceState) {
////        super.onCreate(savedInstanceState);
////        binding = ActivityMain5Binding.inflate(getLayoutInflater());
////        setContentView(binding.getRoot());
////
////        Intent intent = this.getIntent();
////        Batiment batiment = intent.getParcelableExtra("current_batiment");
////
////        Batiment[]
////
////        for (int i = 0; i < batiment.etages.length; i++) {
////            for (int j = 0; j < batiment.etages[i].cameras.length; j++) {
////                Batiment dataEtage_inter = new Batiment("Etage" + Integer.toString(i), "Camera" + Integer.toString(j), Integer.toString(batiment.etages[i].victimes));
////                dataEtageArrayList.add(dataEtage_inter);
////            }
////        }
////
////
////
////        batimentAdapter = new BatimentAdapter(MainActivity5.this, batimentArrayList);
////        binding.ListView.setAdapter(batimentAdapter);
////
////
////
////        binding.ListView.setBackgroundColor(Color.blue(2));
////
////        binding.ListView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
////            @Override
////            public void onItemClick(AdapterView<?> adapterView, View view, int i, long l) {
////                Intent intent = new Intent(MainActivity5.this, MainActivity.class);
////                intent.putExtra("current_batiment", batimentArrayList.get(i));
////                startActivity(intent);
////                finish();
////            }
////        });
////
////    }
////}