package com.example.final_app;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.PointerIcon;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;

import java.util.ArrayList;

public class BatimentAdapter extends ArrayAdapter<Batiment> {

    public BatimentAdapter(@NonNull Context context, ArrayList<Batiment> batiments) {
        super(context, R.layout.layout_list_batiment, batiments);
    }

    @NonNull
    @Override
    public View getView(int position, @Nullable View view, @NonNull ViewGroup parent) {
        Batiment batiment = getItem(position);

        if (view == null) {
            view = LayoutInflater.from(getContext()).inflate(R.layout.layout_list_batiment, parent, false);
        }

        TextView nom_batiment = view.findViewById(R.id.liste_batiment);
        View couleur_batiment = view.findViewById(R.id.Couleur_batiment);

        nom_batiment.setText(batiment.nom);
        if (batiment.danger) {
            couleur_batiment.setBackgroundColor(ContextCompat.getColor(getContext(), android.R.color.holo_red_light));
            // TODO pour régler le probleme du rond qui devient un carré
            // TODO passer par le drawable shapeDrawable.getPaint().setcolors()
        } else {
            couleur_batiment.setBackgroundColor(ContextCompat.getColor(getContext(), android.R.color.holo_green_light));
        }
        return view;
    }
}

