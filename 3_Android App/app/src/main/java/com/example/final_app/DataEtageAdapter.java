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

public class DataEtageAdapter extends ArrayAdapter<DataEtage> {

    public DataEtageAdapter(@NonNull Context context, ArrayList<DataEtage> dataEtages) {
        super(context, R.layout.layout_list_etage, dataEtages);
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