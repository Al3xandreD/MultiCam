package com.example.final_app;
import android.os.Parcel;
import android.os.Parcelable;

public class Batiment implements Parcelable {
    Etage[] etages;
    String nom;
    boolean danger;

    public Batiment(Etage[] etages, String nom) {
        this.etages = etages;
        this.nom = nom;
    }

    // methode pour actualiser l'état dans le batiment
    public void EnDanger(){
        danger = true;
    }

    public int NbEtage() {
        return etages.length;
    }

    ////////////////////////////////////////////
    // Parcelable method
    @Override
    public int describeContents() {
        return 0;
    }

    // Constructor for Parcelable
    protected Batiment(Parcel in) {
        nom = in.readString();
        danger = in.readByte() != 0;
        etages = in.createTypedArray(Etage.CREATOR);
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeString(nom);
        dest.writeByte((byte) (danger ? 1 : 0));
        dest.writeTypedArray(etages, flags);
    }
    // Creator constant for Parcelable
    public static final Creator<Batiment> CREATOR = new Creator<Batiment>() {
        @Override
        public Batiment createFromParcel(Parcel in) {
            return new Batiment(in);
        }

        @Override
        public Batiment[] newArray(int size) {
            return new Batiment[size];
        }
    };
}
