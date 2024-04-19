package com.example.final_app;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.Toast;
import com.example.final_app.databinding.Activity2Binding;
import java.util.ArrayList;

public class Activity2 extends AppCompatActivity {

    Activity2Binding binding;
    DataEtageAdapter dataEtageAdapter;
    ArrayList<DataEtage> dataEtageArrayList = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = Activity2Binding.inflate(getLayoutInflater());

        Intent intent = this.getIntent();
        Batiment batiment = intent.getParcelableExtra("current_batiment");


        for (int i = 0; i < batiment.etages.length; i++) {
            for (int j = 0; j < batiment.etages[i].cameras.length; j++) {
                DataEtage dataEtage_inter = new DataEtage("Etage" + Integer.toString(i), "Camera" + Integer.toString(j), Integer.toString(batiment.etages[i].victimes), R.drawable.logo_carte, R.drawable.logo_camera);
                dataEtageArrayList.add(dataEtage_inter);
            }
        }

        dataEtageAdapter = new DataEtageAdapter(Activity2.this, dataEtageArrayList);
        binding.ListViewActivity2.setAdapter(dataEtageAdapter);

        binding.ListViewActivity2.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> adapterView, View view, int i, long l) {

                // Obtenez l'élément de données à partir de la position cliquée
                Toast.makeText(Activity2.this, "Vous avez cliqué sur ", Toast.LENGTH_SHORT).show();

                DataEtage clickedDataEtage = dataEtageArrayList.get(i);

                // Créez une intention pour démarrer l'ActivityCamera avec des données supplémentaires
                Intent intent_act2 = new Intent(Activity2.this, ActivityCamera.class);
                // Ajoutez des données supplémentaires à l'intention
                intent_act2.putExtra("clicked_data_etage", clickedDataEtage);
                // Démarrez l'activité avec l'intention
                startActivity(intent_act2);
            }
        });
        setContentView(binding.getRoot());
        // le probleme pourrait venir du faut faque d'abord on génère puis après on peuple avec le intent

    }
}
