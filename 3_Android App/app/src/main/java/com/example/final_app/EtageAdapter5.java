package com.example.final_app;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;
import java.util.ArrayList;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;
import java.util.ArrayList;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;

public class EtageAdapter5 extends ArrayAdapter<DataEtage> {

    public EtageAdapter5(@NonNull Context context, ArrayList<DataEtage> dataEtages) {
        super(context, R.layout.layout_list_batiment, dataEtages);
    }

    @NonNull
    @Override
    public View getView(int position, @Nullable View view, @NonNull ViewGroup parent) {
        DataEtage dataEtage = getItem(position);

        if (view == null) {
            view = LayoutInflater.from(getContext()).inflate(R.layout.layout_list_etage, parent, false);
        }

        TextView nom_etage = view.findViewById(R.id.liste_etage);
        TextView nom_camera = view.findViewById(R.id.liste_camera);
        TextView nom_victime = view.findViewById(R.id.liste_victime);
        ImageView logo_carte = view.findViewById(R.id.logoImageCarte);
        ImageView logo_camera = view.findViewById(R.id.logoImageCamera);

        return view;
    }
    // todo : écrire ma propre fonction qu return des positions

}

//package com.example.final_app;
//
//import android.content.Context;
//import android.view.LayoutInflater;
//import android.view.View;
//import android.view.ViewGroup;
//import android.widget.ArrayAdapter;
//import android.widget.TextView;
//
//import androidx.annotation.NonNull;
//import androidx.annotation.Nullable;
//import androidx.core.content.ContextCompat;
//
//import java.util.ArrayList;
//
//public class EtageAdapter5 extends ArrayAdapter<Batiment> {
//
//    public EtageAdapter5(@NonNull Context context, ArrayList<Batiment> batiments) {
//        super(context, R.layout.layout_list_batiment, batiments);
//    }
//
//    @NonNull
//    @Override
//    public View getView(int position, @Nullable View view, @NonNull ViewGroup parent) {
//        Batiment batiment = getItem(position);
//
//        if (view == null) {
//            view = LayoutInflater.from(getContext()).inflate(R.layout.layout_list_batiment, parent, false);
//        }
//
//        TextView nom_batiment = view.findViewById(R.id.liste_batiment);
//        View couleur_batiment = view.findViewById(R.id.Couleur_batiment);
//
//        nom_batiment.setText(batiment.nom);
//        if (batiment.danger) {
//            couleur_batiment.setBackgroundColor(ContextCompat.getColor(getContext(), android.R.color.holo_red_light));
//            // TODO pour régler le probleme du rond qui devient un carré
//            // TODO passer par le drawable shapeDrawable.getPaint().setcolors()
//        } else {
//            couleur_batiment.setBackgroundColor(ContextCompat.getColor(getContext(), android.R.color.holo_green_light));
//        }
//        return view;
//    }
//}
//
