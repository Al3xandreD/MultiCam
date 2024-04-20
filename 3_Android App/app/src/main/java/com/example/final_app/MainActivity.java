package com.example.final_app;

import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;

import androidx.appcompat.app.AppCompatActivity;

import com.example.final_app.databinding.ActivityMainBinding;

import java.util.ArrayList;


public class MainActivity extends AppCompatActivity {

    Etage etages[];
    Etage etage1[];
    Batiment bat1, bat2, bat3;


    /////////// Utilisation de binding car sencé être mieux que de la faire avec de findviewById
    ActivityMainBinding binding;
    BatimentAdapter batimentAdapter;
    ArrayList<Batiment> batimentArrayList = new ArrayList<>();
    Batiment batiment;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());


        //////////////////////////

        etages = new Etage[3];
        etages[0] = new Etage(0);
        etages[1] = new Etage(1);
        etages[2] = new Etage(2);

        etage1 = new Etage[1];
        etage1[0] = new Etage(0);


        bat1 = new Batiment(etages, "Grand Hall");
        bat2 = new Batiment(etages, "Bâtiment F");
        bat1.EnDanger();
        bat3 = new Batiment(etages, "Bâtiment E");

        /////////////////////////

        // fait de manière brute pour l'instant tant qu'on ne sait pas comment seront les données

        batimentArrayList.add(bat1);
        batimentArrayList.add(bat2);
        batimentArrayList.add(bat3);

        ///////////////////////////

        batimentAdapter = new BatimentAdapter(MainActivity.this, batimentArrayList);
        binding.ListView.setAdapter(batimentAdapter);
//        ViewGroup.LayoutParams layoutParams = binding.ListView.getLayoutParams();
//        layoutParams.height = 10;


        binding.ListView.setBackgroundColor(Color.blue(2));

        binding.ListView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> adapterView, View view, int i, long l) {
                Intent intent = new Intent(MainActivity.this, ActivityCamera.class);
                intent.putExtra("current_batiment", batimentArrayList.get(i));
                startActivity(intent);
                //finish();
            }
        });

    }
}